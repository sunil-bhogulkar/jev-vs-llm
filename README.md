# Jev vs LLM — Classifier ("System One") vs. Language Model ("System Two")

A minimal, offline demonstration of the difference between a **Jev-style
classifier** and a traditional **LLM**, using a single classic question:

> **How many times does the letter 'r' appear in the word "strawberry"?**
> (Correct answer: **3**. Strawberry is `s-t-r-a-w-b-e-r-r-y`.)

Jev (TypeSafe AI's "System One" model) is a new class of model that does not
generate text — it takes unstructured state plus typed questions and returns
typed decisions with probabilities, tuned to be fast and calibrated. Because
Jev is gated behind an early-access API key, this project swaps in
**GLiNER2** (`fastino/gliner2.5-small-v1`), a free, fully-local, CPU-only
zero-shot classifier with the same one-job contract: give it text and a fixed
list of labels, get back a label with a confidence score, no prose.

The LLM side runs **deepseek-r1:7b** locally via Ollama — a "System Two"
model that reasons step-by-step and generates free text.

## Why this question

Letter-counting is a deliberate trap that exposes each family's trade-offs:

| | Jev-style classifier | LLM |
|---|---|---|
| **Method** | pattern-matches against 5 labels you predefined | reasons, counts, explains |
| **Speed** | ~100 ms | seconds to minutes (CPU) |
| **Output** | label + confidence, nothing else | free text you must parse |
| **Special superpower** | calibrated guesses, schema-safe, cheap | open-ended reasoning & explanation |

Both can get this wrong — the classifier because it can't actually count, and
the LLM because it can hallucinate or mis-tokenize the word.

## Repository layout

```
jev-vs-llm/
├── classifier/
│   └── classify.py     # GLiNER2 zero-shot classifier, prints one-line JSON
├── llm/
│   └── ask_llm.py      # deepseek-r1:7b via Ollama, prints one-line JSON
├── comparison/
│   └── compare.py      # runs both, renders the comparison table
├── requirements.txt
├── AGENTS.md           # working reference for agents (read this before edits)
└── .gitignore
```

## Setup

Requires Python 3.14+ (the bundled `.venv` works on the machine this was
built on; recreate it elsewhere if needed).

```bash
# 1. Dependencies (torch must be the CPU build)
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install "gliner2[local]" protobuf

# 2. LLM side — install Ollama, then pull the model
ollama pull deepseek-r1:7b   # ensure `ollama serve` is running
```

The GLiNER2 model downloads into the HuggingFace cache on first load
(`fastino/gliner2.5-small-v1`, ~200 MB).

## Running

```bash
.venv/bin/python comparison/compare.py   # full comparison (~90 s; LLM is slow)
```

Individual steps:

```bash
.venv/bin/python classifier/classify.py   # classifier only (fast)
.venv/bin/python llm/ask_llm.py           # LLM only (slow, 1-2 min)
```

## Sample output

```
Word: 'strawberry'  |  True answer: the letter 'r' appears 3 times

| model | answer | correct? | confidence / detail | latency | gave explanation? |
|---|---|---|---|---|---|
| gliner2.5-small (GLiNER2, Jev-style classifier) | 3 | yes | confidence 0.239 | 123.3 ms | no |
| deepseek-r1:7b (Ollama, local LLM) | 1 | no | eval tokens 312 | 91.0 s | yes |
```

## Interpreting results

- The classifier answers in ~100 ms with a **low confidence score** (~0.2–0.3)
  — a tell that it's guessing; it has no mechanism to count, it can only pick
  the most plausible of the labels you supplied, and it can never justify its
  choice.
- The LLM takes ~90 s but **reasons about the word** and can explain its
  reasoning. That reasoning is not guaranteed correct: across runs it has
  answered both `1` and `3`.
- Both are stochastic — re-run a few times to see the variance. This mirrors
  the real-world lesson: use a classifier for fast, structured, low-stakes
  decisions (routing, gating, scoring) and an LLM where reasoning or
  generation is required; Jev-style classifiers are complements to, not
  replacements for, LLMs.

## Limitations / notes

- Jev itself is not used (early-access API key); GLiNER2 stands in for the
  same "typed decisions, no text" contract. Numbers here are illustrative,
  not a benchmark.
- CPU-only inference is slow for the 7B reasoning model; `num_predict` caps
  the LLM output in `ask_llm.py` to keep runs bounded.
- Each script's stdout contract (single-line JSON) is relied on by
  `compare.py` — see `AGENTS.md` before changing it.