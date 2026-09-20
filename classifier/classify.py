import json
import sys
import time

from gliner2 import AutoExtractor

QUESTION = "How many times does the letter 'r' appear in the word 'strawberry'?"

WORDS_TO_N = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

LABELS = [
    "one letter r in the word",
    "two letter r's in the word",
    "three letter r's in the word",
    "four letter r's in the word",
    "five letter r's in the word",
]


def main() -> None:
    t0 = time.perf_counter()
    extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-small-v1")
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = extractor.classify_text(
        QUESTION,
        {"how_many_r": LABELS},
        threshold=0.0,
        include_confidence=True,
    )
    classify_ms = (time.perf_counter() - t0) * 1000

    picked = result["how_many_r"]
    picked_label = picked["label"]
    confidence = picked["confidence"]

    n = WORDS_TO_N[picked_label.split(" ")[0]]

    print(json.dumps({
        "model": "gliner2.5-small (GLiNER2, Jev-style classifier)",
        "question": QUESTION,
        "answer": n,
        "label": picked_label,
        "confidence": round(confidence, 3),
        "correct": n == 3,
        "explained": False,
        "classify_ms": round(classify_ms, 1),
        "model_load_s": round(load_s, 1),
    }))


if __name__ == "__main__":
    sys.exit(main())