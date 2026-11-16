"""Run a prompt version on a few emails, several times each, and report how stable the outputs are.

Usage: uv run python scripts/smoke_test.py [--version v1] [--runs 5]

The emails below were written by an LLM (Claude). They are for smoke testing only
and must never be added to the golden dataset, which has to be human-written.
"""

import argparse
import asyncio
from collections import Counter
from statistics import mean

from dotenv import load_dotenv

from regression_gate.classifier import classify, make_client
from regression_gate.prompts import load_prompt, prompt_path

# (expected category under the labeling policy in DECISIONS.md, email)
SMOKE_EMAILS = [
    ("billing", "Hi, I was charged twice for my monthly subscription on November 3rd. Can you refund one of the charges?"),
    ("technical", "The app crashes every time I try to export a report as PDF. I'm on version 4.2 on Windows 11."),
    ("account", "I forgot my password and the reset email never arrives. I've checked my spam folder."),
    ("general", "Do you offer discounts for non-profit organizations? We're a small charity looking at your product."),
    # Boundary: login is account, but a broken login page is technical.
    ("technical", "When I try to log in I just get a page saying '500 Internal Server Error'. My password is definitely right."),
    # Boundary: involves a payment, but the thing to resolve is a feature not working.
    ("technical", "I upgraded to Pro yesterday and paid, but none of the Pro features are showing up in my dashboard."),
]


async def main(version: str, runs: int) -> None:
    load_dotenv()
    config = load_prompt(prompt_path(version))
    client = make_client()

    category_stable = summary_identical = correct_runs = total_runs = 0
    all_lengths = []

    for expected, email in SMOKE_EMAILS:
        outcomes = await asyncio.gather(*(classify(email, config, client) for _ in range(runs)))
        ok = [o for o in outcomes if o.result]
        categories = Counter(o.result.category.value for o in ok)
        summaries = Counter(o.result.summary for o in ok)
        lengths = [len(o.result.summary) for o in ok]
        all_lengths += lengths

        category_stable += len(categories) == 1 and len(ok) == runs
        summary_identical += len(summaries) == 1 and len(ok) == runs
        correct_runs += categories[expected]
        total_runs += runs

        print(f"\n{email[:75]}...")
        print(f"  expected: {expected:10} got: {dict(categories)}")
        print(f"  distinct summaries: {len(summaries)} of {len(ok)}   lengths: {lengths}")
        for summary, count in summaries.items():
            print(f"    {count}x {summary}")
        for o in outcomes:
            if o.error_type:
                print(f"  ERROR {o.error_type}: {o.error_message}")
        if ok:
            print(f"  latency avg {mean(o.latency_ms for o in ok):.0f} ms   tokens in/out {ok[0].input_tokens}/{ok[0].output_tokens}"
                  f"   served {ok[0].served_model}   normalized {sum(o.category_normalized for o in ok)}")

    n = len(SMOKE_EMAILS)
    print(f"\n=== {version}: {n} emails x {runs} runs ===")
    print(f"correct category:            {correct_runs}/{total_runs} runs")
    print(f"category same across runs:   {category_stable}/{n} emails")
    print(f"summary identical across runs: {summary_identical}/{n} emails")
    print(f"summary length: min {min(all_lengths)}, max {max(all_lengths)}, "
          f"at 95+ chars: {sum(l >= 95 for l in all_lengths)}/{len(all_lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.version, args.runs))
