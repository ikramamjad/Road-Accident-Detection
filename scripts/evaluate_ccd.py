"""
Benchmark Evaluation Script.
Runs stratified evaluations across Day/Night, Clear/Rain/Fog, Sparse/Dense IDD traffic,
and benchmarks against published Car Crash Dataset (CCD) literature.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from road_accident_detection.evaluation.stratified_eval import StratifiedEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Accident Detection Pipeline on CCD Benchmark")
    parser.add_argument("--stratified", action="store_true", default=True, help="Report stratified metrics")
    args = parser.parse_args()

    evaluator = StratifiedEvaluator()
    print("[Evaluation] Running evaluation on stratified test distribution (CCD + IDD Traffic)...")
    report = evaluator.evaluate_benchmark()

    markdown_summary = evaluator.format_report_markdown(report)
    print("\n" + markdown_summary)


if __name__ == "__main__":
    main()
