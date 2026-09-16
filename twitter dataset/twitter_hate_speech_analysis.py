"""Reproducible ITCS444 Project 1 analysis with no third-party ML packages.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data") / "twitter_parsed_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_assets"
RANDOM_STATE = 444
CLASS_ORDER = ["none", "racism", "sexism"]
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|URL|USER", re.IGNORECASE)


def clean_tweet(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def stratified_split(rows: list[dict], test_fraction: float = 0.20) -> tuple[list[dict], list[dict]]:
    rng = random.Random(RANDOM_STATE)
    train, test = [], []
    for label in CLASS_ORDER:
        group = [row for row in rows if row["Annotation"] == label]
        rng.shuffle(group)
        cut = round(len(group) * (1 - test_fraction))
        train.extend(group[:cut])
        test.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def train_naive_bayes(rows: list[dict]):
    document_counts = Counter(row["Annotation"] for row in rows)
    token_counts = {label: Counter() for label in CLASS_ORDER}
    token_totals = Counter()
    vocabulary = set()
    for row in rows:
        tokens = tokenise(row["clean_text"])
        token_counts[row["Annotation"]].update(tokens)
        token_totals[row["Annotation"]] += len(tokens)
        vocabulary.update(tokens)
    return document_counts, token_counts, token_totals, len(vocabulary)


def predict(text: str, model) -> str:
    document_counts, token_counts, token_totals, vocab_size = model
    total_documents = sum(document_counts.values())
    scores = {}
    for label in CLASS_ORDER:
        score = math.log(document_counts[label] / total_documents)
        denominator = token_totals[label] + vocab_size
        for token in tokenise(text):
            score += math.log((token_counts[label][token] + 1) / denominator)
        scores[label] = score
    return max(scores, key=scores.get)


def evaluate(test: list[dict], predictions: list[str]) -> dict:
    matrix = {actual: {predicted: 0 for predicted in CLASS_ORDER} for actual in CLASS_ORDER}
    for row, predicted in zip(test, predictions):
        matrix[row["Annotation"]][predicted] += 1
    per_class, f1s, supports = {}, [], []
    for label in CLASS_ORDER:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in CLASS_ORDER if actual != label)
        fn = sum(matrix[label][predicted] for predicted in CLASS_ORDER if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        support = sum(matrix[label].values())
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": support}
        f1s.append(f1); supports.append(support)
    accuracy = sum(matrix[label][label] for label in CLASS_ORDER) / len(test)
    return {"accuracy": round(accuracy, 4), "macro_f1": round(sum(f1s) / len(f1s), 4),
            "weighted_f1": round(sum(f * s for f, s in zip(f1s, supports)) / sum(supports), 4),
            "per_class": per_class, "confusion_matrix": matrix}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open(encoding="utf-8", newline="") as file:
        raw = list(csv.DictReader(file))
    rows = []
    for row in raw:
        if row.get("Text") and row.get("Annotation") in CLASS_ORDER:
            row["clean_text"] = clean_tweet(row["Text"])
            rows.append(row)
    counts = Counter(row["Annotation"] for row in rows)
    feature_rates = {}
    for label in CLASS_ORDER:
        group = [row for row in rows if row["Annotation"] == label]
        feature_rates[label] = {
            "mean_characters": round(sum(len(row["Text"]) for row in group) / len(group), 2),
            "mean_words": round(sum(len(tokenise(row["clean_text"])) for row in group) / len(group), 2),
            "url_percent": round(sum(bool(re.search(r"https?://", row["Text"])) for row in group) / len(group) * 100, 2),
            "mention_percent": round(sum(bool(re.search(r"@\w+", row["Text"])) for row in group) / len(group) * 100, 2),
        }
    train, test = stratified_split(rows)
    model = train_naive_bayes(train)
    predictions = [predict(row["clean_text"], model) for row in test]
    summary = {
        "raw_rows": len(raw), "usable_rows": len(rows), "removed_incomplete_rows": len(raw) - len(rows),
        "class_counts": {label: counts[label] for label in CLASS_ORDER},
        "class_percentages": {label: round(counts[label] / len(rows) * 100, 2) for label in CLASS_ORDER},
        "feature_rates_by_class": feature_rates,
        "baseline": {"model": "Multinomial Naive Bayes with tokenized normalized tweet text and Laplace smoothing",
                     "train_rows": len(train), "test_rows": len(test), **evaluate(test, predictions)},
    }
    with (OUTPUT_DIR / "cleaned_twitter_dataset.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Text", "Annotation", "oh_label"])
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in writer.fieldnames} for row in rows)
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
