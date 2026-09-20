import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
CLASSIFIER = ROOT / "classifier" / "classify.py"
LLM = ROOT / "llm" / "ask_llm.py"


def run_script(path: Path) -> dict:
    proc = subprocess.run(
        [str(VENV_PY), str(path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{path.name} failed with exit code {proc.returncode}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def render_row(r: dict) -> str:
    if "classify_ms" in r:
        speed = f"{r['classify_ms']} ms"
        conf = f"confidence {r['confidence']}"
    else:
        speed = f"{r['latency_s']} s"
        conf = f"eval tokens {r['eval_tokens']}"
    return (
        f"| {r['model']} | {r['answer']} | {('yes' if r['correct'] else 'no')} | "
        f"{conf} | {speed} | {'yes' if r['explained'] else 'no'} |"
    )


def main() -> None:
    print("Running Jev-style classifier...")
    class_res = run_script(CLASSIFIER)
    print("Running local LLM (this is the slow one)...")
    llm_res = run_script(LLM)

    print()
    print("Word: 'strawberry'  |  True answer: the letter 'r' appears 3 times")
    print()
    print("| model | answer | correct? | confidence / detail | latency | gave explanation? |")
    print("|---|---|---|---|---|---|")
    print(render_row(class_res))
    print(render_row(llm_res))

    print()
    print("Takeaway:")
    print(" - Classifier (System One / Jev-style): guesses from a fixed list,")
    print("   no way to actually 'count' — few labels are even plausible.")
    print(" - LLM (System Two): can reason, justify itself, and say HOW, but")
    print("   is ~orders of magnitude slower and can still hallucinate.")


if __name__ == "__main__":
    sys.exit(main())