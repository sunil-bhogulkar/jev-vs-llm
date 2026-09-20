# AGENTS.md

Reference for working on this project: a side-by-side comparison of a
Jev-style zero-shot classifier ("System One") vs. a local LLM ("System Two")
on the same question, using `strawberry` letter-counting as the example.

## Project layout

```
strawberry-compare/
├── classifier/
│   └── classify.py     # Jev-style classifier (GLiNER2), prints one-line JSON
├── llm/
│   └── ask_llm.py      # local LLM via Ollama (deepseek-r1:7b), prints one-line JSON
├── comparison/
│   └── compare.py      # runs both, parses their JSON, prints a comparison table
├── requirements.txt
└── AGENTS.md
```

## Environment / setup

- Python 3.14 venv at `.venv/` (already created). Do not recreate lightly.
- Classifier deps are CPU builds and picky:
  - `torch` MUST come from the CPU index: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - `gliner2[local]` plus `protobuf` (DeBERTa fast tokenizer needs it — install fails confusingly without it).
  - Model `fastino/gliner2.5-small-v1` is cached under the HuggingFace cache (~200 MB).
- LLM side needs **Ollama running** (`ollama serve`) with `deepseek-r1:7b` pulled. There is NO pip dependency for the LLM; it talks to `http://localhost:11434` via stdlib `urllib`.

## How to run

Everything runs through the shared venv interpreter:

```
.venv/bin/python comparison/compare.py
```

- Takes ~90s+ because `deepseek-r1` is a reasoning model running on CPU.
- Individual steps:
  - `.venv/bin/python classifier/classify.py`
  - `.venv/bin/python llm/ask_llm.py`

## Output contract (do not break)

`classify.py` and `ask_llm.py` each print a SINGLE JSON object on the last
stdout line; `compare.py` shells out to them, takes the last non-empty line,
and `json.loads` it. Keep that contract when editing:
- classifier keys: `model, question, answer, label, confidence, correct, explained, classify_ms, model_load_s`
- llm keys: `model, question, answer, raw_answer_text, reasoning_len, correct, explained, latency_s, eval_tokens`

`answer` is documented under `correct` ("letter-r count of strawberry is **3**").

## Gotchas observed while building

- `classify.py` parses its own label text back to an int (`WORDS_TO_N`); label
  wording and that map must stay in sync. Labels are `"one/two/three/four/five letter r's in the word"`.
- Both models are stochastic per run: the classifier's picked label varies run
  to run and its confidence stays low (~0.2-0.3), and the LLM has answered
  both 1 and 3 across runs. Results are illustrative, not stable.
- `classifier/classify.py` suppresses nothing but is noisy: GLiNER2 prints
  UserWarnings about legacy tokenizer metadata, torch.jit script, DeBERTa
  attn fallback. Run via `compare.py` these leak to stderr unless suppressed;
  results are still printed on stdout.
- Model load takes a few seconds on the classifier; timing in `classify_ms` is
  inference-only (load separated as `model_load_s`). Latency report for model
  load happens on first call only.
- GLiNER2 `classify_text` `threshold=0.0` forces a pick (like Jev's argmax);
  default 0.5 may return nothing when confidence is low.
- Ollama `/api/generate` with `stream:false` only returns once generation
  completes, so the first LLM call after idle can "hang" many minutes while the
  model loads. `num_predict` caps output; keep it small for quick tests.

## Commands / conventions

- One-off QA: `curl http://localhost:11434/api/tags` to confirm Ollama is up.
- Inference flags used by the tree, keep consistent when changing model size.