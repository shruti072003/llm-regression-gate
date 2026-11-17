"""Show how the golden dataset is spread across categories, difficulty and tags.

Usage: uv run python scripts/dataset_stats.py
"""

from collections import Counter

from regression_gate.dataset import Difficulty, EdgeTag, load_dataset
from regression_gate.schemas import Category

TARGET_PER_CATEGORY = 15

dataset = load_dataset()
cases = dataset.cases
print(f"golden dataset v{dataset.version} ({dataset.content_hash}): {len(cases)} cases\n")

by_category = Counter(c.expected_category for c in cases)
print("category       cases   (target ~15)")
for category in Category:
    n = by_category[category]
    print(f"  {category.value:12} {n:5}   {'#' * n}{'.' * max(0, TARGET_PER_CATEGORY - n)}")

by_difficulty = Counter(c.difficulty for c in cases)
print("\ndifficulty")
for level in Difficulty:
    print(f"  {level.value:12} {by_difficulty[level]:5}")

by_tag = Counter(tag for c in cases for tag in c.tags)
print("\nedge-case tags")
for tag in EdgeTag:
    print(f"  {tag.value:16} {by_tag[tag]:3}{'   <- none yet' if not by_tag[tag] else ''}")
