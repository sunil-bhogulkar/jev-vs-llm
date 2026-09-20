import json
import re
import sys
import time
import urllib.request

MODEL = "deepseek-r1:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"
QUESTION = "How many times does the letter 'r' appear in the word 'strawberry'?"

PROMPT = (
    "How many times does the letter 'r' appear in the word 'strawberry'?\n"
    "Think it through step by step, then state the final answer as a single number."
)


def strip_reasoning(text: str) -> tuple[str, str]:
    reasoning = ""
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
        text = text.replace(m.group(0), "").strip()
    return text, reasoning


def first_number(text: str) -> int | None:
    for tok in re.findall(r"\d+", text):
        return int(tok)
    return None


def main() -> None:
    payload = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "num_predict": 400,
        "temperature": 0.2,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    latency_s = time.perf_counter() - t0

    raw = data.get("response", "")
    answer_text, reasoning = strip_reasoning(raw)
    n = first_number(answer_text)

    print(json.dumps({
        "model": f"{MODEL} (Ollama, local LLM)",
        "question": QUESTION,
        "answer": n,
        "raw_answer_text": answer_text[:200],
        "reasoning_len": len(reasoning),
        "correct": n == 3,
        "explained": n is not None and len(reasoning or answer_text) > 1,
        "latency_s": round(latency_s, 1),
        "eval_tokens": data.get("eval_count"),
    }))


if __name__ == "__main__":
    sys.exit(main())