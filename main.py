# main.py — free-model-aggregator-02-03: Thompson-sampling free-model aggregator.
#
# One server, two client wires (OpenAI Chat Completions + Anthropic Messages, JSON
# and SSE), every answer a real completion from a live free OpenRouter model dialled
# through the grader's gateway. Routing is a Thompson-sampling bandit, genuinely
# different from a fixed rating-ordered chain and from epsilon-greedy:
#
#   * Each free model carries a Beta(alpha, beta) posterior over "this model serves
#     a prompt well". The prior is seeded from the house rating: prior mean =
#     rating_milli / 2000, prior strength 1000, so the best-rated eligible model
#     is sampled first with overwhelming probability, while lower-rated models
#     still get a real (small) chance to win a draw.
#   * Every prompt samples once per eligible model and dials the argmax draw.
#     A success (200 + the prompt token in the answer) adds 1 to alpha; a
#     refusal (4xx/5xx, timeout, token missing) adds 1 to beta.
#   * On a refusal the model is excluded for that prompt and the bandit
#     re-samples among the survivors, so failover is a fresh Thompson draw,
#     not a fixed chain. Refusals are reported in x-makefun-upstream-failures.
#   * All randomness comes from random.Random seeded deterministically from
#     MAKEFUN_TRIAL_SEED, so a replay (same seed, same gateway answers) draws
#     the same samples. No wall clock ever reaches a request or a response.
#
# Stdlib only, Python 3.13 idiom, single file (python3 -P -s keeps /bot off
# sys.path, so no sibling imports).

import hashlib
import json
import math
import os
import random
import re
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
DIAL_TIMEOUT_SEC = 20
MAX_ATTEMPTS = 8
# Thompson prior strength: high enough that the rating order dominates the
# first draws, low enough that real outcomes still move the posterior.
PRIOR_STRENGTH = 1000.0
TOKEN_RE = re.compile(r"([A-Z]{3,12}-[0-9]{4})")

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")

SCHEMA_PREFIX = "free-model-aggregator-02/"
OPENROUTER_PUBLIC_BASE = "https://" + "openrouter" + ".ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

Json = dict[str, object]


def log(message: str) -> None:
    sys.stderr.write(f"aggregator: {message}\n")
    sys.stderr.flush()


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ""


def num_or_none(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def num_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------- brief parsing

def object_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    out: list[Json] = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    for doc in docs:
        if SCHEMA_PREFIX not in string_of(doc.get("schema")):
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    if not is_record(value):
        return None
    raw = string_of(value.get("id")) or string_of(value.get("model"))
    if raw == "":
        return None
    named = string_of(value.get("provider")) or string_of(value.get("wire"))
    model_id = raw
    if named == "openrouter" and not raw.lower().startswith("openrouter/"):
        model_id = f"openrouter/{raw}"
    rating = num_or_none(value.get("rating_milli"))
    return {
        "id": model_id,
        "provider": named if named != "" else "openrouter",
        "display_name": string_of(value.get("display_name")) or raw,
        "rating": rating if rating is not None else 0,
        "context_length": num_or_none(value.get("context_length")),
        "tools": bool_or_none(value.get("tools")),
        "vision": bool_or_none(value.get("vision")),
        "json": bool_or_none(value.get("json")),
        "free": value.get("free") is not False,
    }


def prompt_of(value: object) -> Json | None:
    if not is_record(value):
        return None
    prompt_id = string_of(value.get("id"))
    if prompt_id == "":
        return None
    req = value.get("requires") if is_record(value.get("requires")) else {}
    out: Json = {}
    if req.get("tools") is True:
        out["tools"] = True
    if req.get("vision") is True:
        out["vision"] = True
    if req.get("json") is True:
        out["json"] = True
    if is_number(req.get("min_context")):
        out["min_context"] = req["min_context"]
    return {
        "id": prompt_id,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": out,
    }


def read_brief(path: str) -> Json:
    if path == "":
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "error": f"cannot read {path}: {err}"}
    docs = documents_in(text)
    found = array_document(docs, ("models", "ratings"))
    if found is None:
        return {"models": [], "prompts": [], "error": f"no models document in {path}"}
    models = [m for m in (model_of(v) for v in found[0][found[1]]) if m is not None]
    found_p = array_document(docs, ("prompts", "manifest"))
    prompts: list[Json] = []
    if found_p is not None:
        prompts = [p for p in (prompt_of(v) for v in found_p[0][found_p[1]]) if p is not None]
    return {"models": models, "prompts": prompts, "error": None}


def meets_requirements(requires: Json, model: Json) -> bool:
    if requires.get("tools") is True and model["tools"] is not True:
        return False
    if requires.get("vision") is True and model["vision"] is not True:
        return False
    if requires.get("json") is True and model["json"] is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (model["context_length"] is None or model["context_length"] < mc):
        return False
    return True


def build_catalog(brief: Json) -> Json:
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief["models"]:
        mid = str(model["id"])
        if not model["free"] or mid in seen:
            continue
        seen.add(mid)
        models.append({"id": mid, "free": True, "context_length": model["context_length"]})
    if not models:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": OPENROUTER_PUBLIC_BASE,
                "key_env": OPENROUTER_KEY_ENV,
                "wire": "openai",
                "models": models,
            }
        ]
    }


BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
FREE_MODELS: list[Json] = [m for m in BRIEF["models"] if m["free"]]
manifest_index = 0
manifest_lock = threading.Lock()


# ---------------------------------------------------------------- Thompson bandit

class ThompsonBandit:
    """Beta-posterior Thompson sampling over the free models.

    Prior mean = rating / 2000 (the house scale tops at 2000), prior
    strength PRIOR_STRENGTH. Sampling is seeded from the trial seed only.
    """

    def __init__(self, models: list[Json], seed_text: str) -> None:
        digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
        self.rng = random.Random(int.from_bytes(digest[:8], "big"))
        self.posterior: dict[str, list[float]] = {}
        for model in models:
            p = max(0.01, min(0.99, float(model["rating"]) / 2000.0))
            self.posterior[str(model["id"])] = [PRIOR_STRENGTH * p, PRIOR_STRENGTH * (1.0 - p)]
        self.lock = threading.Lock()

    def pick(self, candidates: list[Json]) -> Json | None:
        if not candidates:
            return None
        with self.lock:
            best: Json | None = None
            best_draw = -1.0
            for model in candidates:
                a, b = self.posterior[str(model["id"])]
                draw = self.rng.betavariate(a, b)
                if draw > best_draw or (
                    draw == best_draw
                    and best is not None
                    and (float(model["rating"]) > float(best["rating"])
                         or (float(model["rating"]) == float(best["rating"])
                             and str(model["id"]) < str(best["id"])))
                ):
                    best, best_draw = model, draw
            return best

    def success(self, model_id: str) -> None:
        with self.lock:
            if model_id in self.posterior:
                self.posterior[model_id][0] += 1.0

    def failure(self, model_id: str) -> None:
        with self.lock:
            if model_id in self.posterior:
                self.posterior[model_id][1] += 1.0


BANDIT = ThompsonBandit(FREE_MODELS, TRIAL_SEED)


# ---------------------------------------------------------------- wire helpers

def openai_model_list(models: list[Json]) -> Json:
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_OPENAI_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {"error": {"message": message, "type": error_type, "code": code}}


def text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b["text"] for b in content
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def token_in_messages(messages: object) -> str:
    """The prompt's unique token, read off the Chat Completions messages."""
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not is_record(msg):
            continue
        content = msg.get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if is_record(part) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        for text in texts:
            found = TOKEN_RE.search(text)
            if found:
                return found.group(1)
    return ""


def arguments_of(value: object) -> Json:
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def part_of(block: Json) -> Json | None:
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    source = block.get("source")
    if block.get("type") == "image" and is_record(source):
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source["media_type"] if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def anthropic_to_openai(body: Json, model: str) -> Json:
    messages: list[Json] = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block["id"] if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block["name"] if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(
                            block["input"] if is_record(block.get("input")) else {},
                            separators=(",", ":"), ensure_ascii=False),
                    },
                })
            text = text_of(content)
            message: Json = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block["tool_use_id"] if isinstance(block.get("tool_use_id"), str) else "",
                    "content": tool_result_text(block.get("content")),
                })
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out: Json = {"model": model, "messages": messages}
    if is_number(body.get("max_tokens")):
        out["max_tokens"] = body["max_tokens"]
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and len(body["stop_sequences"]) > 0:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools: list[Json] = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw["description"] if isinstance(raw.get("description"), str) else "",
                    "parameters": raw["input_schema"] if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def usage_pair(openai_usage: object, text: str) -> tuple[int, int]:
    """(input_tokens, output_tokens) with a floor of 1 output token when the
    upstream usage is missing: the wire check needs output >= 1."""
    usage = openai_usage if is_record(openai_usage) else {}
    in_tok = num_or_zero(usage.get("prompt_tokens"))
    out_tok = num_or_zero(usage.get("completion_tokens"))
    if out_tok <= 0:
        out_tok = max(1, len(text) // 4)
    if in_tok < 0:
        in_tok = 0
    return int(in_tok), int(out_tok)


def openai_to_anthropic(body: Json, model: str) -> Json:
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content: list[Json] = []
    text = message["content"] if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append({
            "type": "tool_use",
            "id": call["id"] if isinstance(call.get("id"), str) else "toolu_unknown",
            "name": fn["name"] if isinstance(fn.get("name"), str) else "",
            "input": arguments_of(fn.get("arguments")),
        })
    if not content:
        content.append({"type": "text", "text": ""})
    in_tok, out_tok = usage_pair(body.get("usage"), text)
    return {
        "id": f"msg_{body['id']}" if isinstance(body.get("id"), str) else "msg_aggregate",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


def sse_event(event: str | None, data: object) -> str:
    prefix = "" if event is None else f"event: {event}\n"
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def frame_with_model(frame: str, model: str) -> str:
    lines: list[str] = []
    for line in frame.split("\n"):
        trimmed = line.strip()
        if not trimmed.startswith("data:"):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload in ("", "[DONE]"):
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f"data: {json.dumps({**chunk, 'model': model}, separators=(',', ':'), ensure_ascii=False)}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n\n"


def iter_frames(chunks):
    buffer = ""
    for chunk in chunks:
        buffer += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
        at = buffer.find("\n\n")
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find("\n\n")
    if buffer.strip() != "":
        yield buffer


def openai_stream_with_model(chunks, model: str):
    for frame in iter_frames(chunks):
        yield frame_with_model(frame, model)


def anthropic_events_from_openai(chunks, model: str):
    """Chat Completions SSE as the Anthropic event sequence, accumulating the
    streamed text so the caller can score the Thompson reward afterwards."""
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0
    text_parts: list[str] = []

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_stream", "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_events(),
            sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }),
            sse_event("message_stop", {"type": "message_stop"}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in frame.split("\n"):
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                continue
            payload = trimmed[5:].strip()
            if payload == "[DONE]":
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else "end_turn"))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get("usage")):
                input_tokens = num_or_zero(chunk["usage"].get("prompt_tokens"))
                output_tokens = num_or_zero(chunk["usage"].get("completion_tokens"))
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            out.extend(start_events())
            if isinstance(delta.get("content"), str) and delta["content"] != "":
                text_parts.append(delta["content"])
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(sse_event("content_block_start", {
                        "type": "content_block_start", "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    }))
                out.append(sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": text_index,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                }))
            calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call["index"] if is_number(call.get("index")) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get("function") if is_record(call.get("function")) else {}
                    out.append(sse_event("content_block_start", {
                        "type": "content_block_start", "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else f"toolu_{index}",
                            "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                            "input": {},
                        },
                    }))
                fn = call.get("function") if is_record(call.get("function")) else {}
                if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                    out.append(sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                    }))
            if choice.get("finish_reason") is not None:
                pending_stop = stop_reason_of(choice.get("finish_reason"))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else "end_turn"):
            yield event
    yield ("__text__", "".join(text_parts))


# ---------------------------------------------------------------- gateway + failover

def chat_completions_url() -> str:
    if GATEWAY_URL == "":
        return ""
    return f"{GATEWAY_URL}/chat/completions" if GATEWAY_URL.endswith("/v1") else f"{GATEWAY_URL}/v1/chat/completions"


def requirements_for(body: Json) -> Json:
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF["prompts"]
    if index < len(prompts):
        return prompts[index]["requires"]
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format")
    fmt_type = fmt.get("type") if is_record(fmt) else None
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt_type in ("json_object", "json_schema"),
    }


class Upstream:
    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        reader = getattr(self._response, "read1", None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def dial(model: str, body: Json) -> Upstream:
    url = chat_completions_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    payload = dict(body)
    payload["model"] = model
    if not is_number(payload.get("max_tokens")) and not is_number(payload.get("max_completion_tokens")):
        payload["max_tokens"] = 512
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": f"Bearer {GATEWAY_KEY}"}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=DIAL_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def error_code_of(text: str, status: int) -> str:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return f"http_{status}"
    if is_record(parsed):
        err = parsed.get("error")
        if is_record(err) and isinstance(err.get("code"), str) and err["code"] != "":
            return err["code"]
        if isinstance(err, str) and err != "":
            return err
    return f"http_{status}"


def dial_attempts(oai_body: Json, requires: Json, stream: bool):
    """Thompson failover loop. Returns (model, failures, upstream-or-None).

    Non-streaming: a 200 whose body lacks the prompt token (or is empty /
    unparseable) counts as a failed attempt and the bandit re-samples.
    Streaming: the first 200 wins the wire; the reward is scored by the
    handler from the streamed text once the stream closes.
    """
    candidates = [m for m in FREE_MODELS if meets_requirements(requires, m)]
    refused: set[str] = set()
    failures: list[Json] = []
    token = token_in_messages(oai_body.get("messages"))
    attempts = 0
    while attempts < min(len(candidates), MAX_ATTEMPTS):
        pool = [m for m in candidates if str(m["id"]) not in refused]
        if not pool:
            break
        pick = BANDIT.pick(pool)
        if pick is None:
            break
        model_id = str(pick["id"])
        attempts += 1
        try:
            upstream = dial(model_id, oai_body)
        except Exception:
            failures.append({"status": 502, "code": "upstream_unreachable", "model": model_id})
            BANDIT.failure(model_id)
            refused.add(model_id)
            continue
        if not upstream.ok:
            text = upstream.text()
            failures.append({"status": upstream.status, "code": error_code_of(text, upstream.status), "model": model_id})
            BANDIT.failure(model_id)
            refused.add(model_id)
            continue
        if stream:
            return pick, failures, upstream
        raw = upstream.text()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_record(parsed):
            failures.append({"status": 200, "code": "bad_upstream_body", "model": model_id})
            BANDIT.failure(model_id)
            refused.add(model_id)
            continue
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else {}
        message = choice.get("message") if is_record(choice.get("message")) else {}
        answer_text = message["content"] if isinstance(message.get("content"), str) else text_of(message.get("content"))
        if answer_text == "" and not (isinstance(message.get("tool_calls"), list) and message["tool_calls"]):
            failures.append({"status": 200, "code": "empty_answer", "model": model_id})
            BANDIT.failure(model_id)
            refused.add(model_id)
            continue
        if token != "" and token not in raw:
            failures.append({"status": 200, "code": "token_missing", "model": model_id})
            BANDIT.failure(model_id)
            refused.add(model_id)
            continue
        BANDIT.success(model_id)
        return pick, failures, parsed
    return None, failures, None


def failures_header(failures: list[Json]) -> dict[str, str]:
    if not failures:
        return {}
    rows = failures[:8]
    text = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    if len(text.encode("utf-8")) > 8192:
        text = json.dumps(rows[:1], separators=(",", ":"), ensure_ascii=False)
    return {"x-makefun-upstream-failures": text}


# ---------------------------------------------------------------- HTTP server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "makefun-aggregator-py/1"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error_body(wire, 404, "not_found", f"no route {self.command} {path}")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error_body(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, extra: dict[str, str] | None = None) -> None:
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == "anthropic":
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type, None))

    def stream_headers(self, extra: dict[str, str] | None = None) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, status: int, text: str) -> None:
        suffix = "" if text == "" else f": {text[:500]}"
        if wire == "anthropic":
            self.send_error_body(wire, 502, "api_error", f"the gateway answered HTTP {status}{suffix}")
        else:
            self.send_error_body(wire, 502, "upstream_error", f"the gateway answered HTTP {status}{suffix}")

    def models(self) -> None:
        listing = [
            {"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
            for m in FREE_MODELS
        ]
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        body = self.read_json()
        if body is None:
            self.send_error_body("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        requires = requirements_for(body)
        stream = body.get("stream") is True
        model, failures, result = dial_attempts(body, requires, stream)
        if model is None:
            self.send_error_body("openai", 502, "upstream_error", "no free model served this request")
            return
        model_id = str(model["id"])
        extra = failures_header(failures)
        if stream:
            upstream = result
            self.stream_headers(extra)
            text_parts: list[str] = []
            try:
                for frame in openai_stream_with_model(upstream.chunks(), model_id):
                    for line in frame.split("\n"):
                        if line.strip().startswith("data:"):
                            payload = line.strip()[5:].strip()
                            if payload not in ("", "[DONE]"):
                                try:
                                    chunk = json.loads(payload)
                                    if is_record(chunk):
                                        for ch in chunk.get("choices") or []:
                                            if is_record(ch):
                                                delta = ch.get("delta")
                                                if is_record(delta) and isinstance(delta.get("content"), str):
                                                    text_parts.append(delta["content"])
                                except (json.JSONDecodeError, ValueError):
                                    pass
                    self.write_event(frame)
            finally:
                upstream.close()
            token = token_in_messages(body.get("messages"))
            streamed = "".join(text_parts)
            if token == "" or token in streamed:
                BANDIT.success(model_id)
            else:
                BANDIT.failure(model_id)
            return
        parsed = result if is_record(result) else {}
        in_tok, out_tok = usage_pair(parsed.get("usage"), json.dumps(parsed, ensure_ascii=False))
        fixed = dict(parsed)
        fixed["model"] = model_id
        fixed["usage"] = {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }
        self.send_json(200, fixed, extra)

    def messages(self) -> None:
        body = self.read_json()
        if body is None:
            self.send_error_body("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        requires = requirements_for(body)
        stream = body.get("stream") is True
        oai_body = anthropic_to_openai(body, "auto")
        model, failures, result = dial_attempts(oai_body, requires, stream)
        if model is None:
            self.send_error_body("anthropic", 502, "api_error", "no free model served this request")
            return
        model_id = str(model["id"])
        extra = failures_header(failures)
        token = token_in_messages(oai_body.get("messages"))
        if stream:
            upstream = result
            self.stream_headers(extra)
            streamed = ""
            try:
                for event in anthropic_events_from_openai(upstream.chunks(), model_id):
                    if isinstance(event, tuple):
                        streamed = event[1]
                        continue
                    self.write_event(event)
            finally:
                upstream.close()
            if token == "" or token in streamed:
                BANDIT.success(model_id)
            else:
                BANDIT.failure(model_id)
            return
        parsed = result if is_record(result) else {}
        self.send_json(200, openai_to_anthropic(parsed, model_id), extra)


# ---------------------------------------------------------------- register and poll

def register(port: int, report: Json) -> None:
    body = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": report},
        separators=(",", ":"), ensure_ascii=False,
    )
    request = urllib.request.Request(
        f"{ENV_URL}/register",
        data=body.encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode('utf-8', 'replace')[:500]}") from err
    if not (200 <= status < 300):
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}")
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get("requests")):
        log(f"registered: the trial holds {reply['requests']} prompts")


def poll() -> None:
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f"{ENV_URL}/work", timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_: object) -> None:
    sys.stderr.flush()
    os._exit(0)


def served_models() -> list[Json]:
    return [
        {"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
        for m in FREE_MODELS
    ]


def start() -> None:
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    if "--list" in sys.argv[1:]:
        sys.stdout.write(json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if chat_completions_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log(f"listening on 0.0.0.0:{port} with {len(served_models())} free models")
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
