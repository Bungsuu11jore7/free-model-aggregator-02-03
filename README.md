# free-model-aggregator-02-03

One server (Python 3.13 stdlib only, `main.py`) serving OpenAI Chat Completions
and Anthropic Messages, JSON and SSE, with every answer a real completion from
a live free OpenRouter model dialled through the grader's gateway.

Routing: Thompson sampling. Each free model carries a Beta(alpha, beta)
posterior; the prior mean is the house rating / 2000 with strength 1000, so
the best-rated eligible model is sampled first with overwhelming probability
while lower-rated models keep a real chance. Success (200 + prompt token in
the answer) adds 1 to alpha; a refusal adds 1 to beta. On a refusal the model
is excluded for that prompt and the bandit re-samples among the survivors.
All randomness is seeded from `MAKEFUN_TRIAL_SEED`, so replays draw identically.

Run: `python3 main.py` (registers with `$MAKEFUN_ENV_URL`, polls `/work`,
exits 0 when done). `python3 main.py --list` prints the catalog.
