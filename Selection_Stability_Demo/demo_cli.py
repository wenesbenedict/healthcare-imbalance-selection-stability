"""Command-line runner for the synthetic selection-stability demonstrator."""
from __future__ import annotations

import argparse
from pathlib import Path

from demo_core import VERSION, DemoConfig, feasibility_table, run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-size", type=int, choices=[100, 250, 500], default=250)
    parser.add_argument("--imbalance-ratio", type=int, choices=[2, 5, 10], default=5)
    parser.add_argument("--noise", type=int, choices=[0, 5, 10], default=5, help="Training label noise percent")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tie-tolerance", type=float, default=0.005)
    parser.add_argument("--threshold-mode", choices=["training_f1", "fixed_0.5"], default="training_f1")
    parser.add_argument("--feasibility-table", action="store_true")
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--output", type=Path, default=Path("demo_summary.csv"))
    args = parser.parse_args()

    if args.feasibility_table:
        print(feasibility_table().to_string(index=False))
        return
    if args.repeats < 100:
        print("WARNING: Fewer than 100 repeats gives wide intervals; use this for demonstration only.")
    config = DemoConfig(
        training_size=args.training_size,
        imbalance_ratio=args.imbalance_ratio,
        noise_percent=args.noise,
        repeats=args.repeats,
        seed=args.seed,
        tie_tolerance=args.tie_tolerance,
        threshold_mode=args.threshold_mode,
    )
    result = run_demo(config)
    print("\nSYNTHETIC SELECTION-STABILITY DEMONSTRATION")
    print("=" * 47)
    print(f"Training size: {config.training_size}")
    print(f"Training minority:majority ratio: 1:{config.imbalance_ratio}")
    print(f"Approximate minority examples per sample: {result['minority_count']}")
    print(f"Simulated training-label noise: {config.noise_percent}%")
    print(f"Repeated samples: {config.repeats}")
    print(f"Selection agreement: {result['agreement']:.3f}")
    print(f"Selection agreement Wilson 95% interval: {result['agreement_95_interval']}")
    print(f"Normalised selection entropy: {result['normalized_entropy']:.3f}")
    print(f"Normalised entropy bootstrap 95% interval: {result['normalized_entropy_95_interval']}")
    print(f"Fixed reference selected by pooled inner CV: {result['fixed_strategy']}")
    print(f"Mean clean-test AP (per-resample selected strategy): {result['mean_selected_test_ap']:.3f}")
    print(f"Mean clean-test AP (fixed reference): {result['mean_fixed_test_ap']:.3f}")
    print(f"Fixed-reference gap (fixed minus selected): {result['fixed_reference_gap']:+.3f}")
    print(f"Fixed-reference gap paired bootstrap 95% interval: {result['fixed_reference_gap_95_interval']}")
    print(f"Tie-break applied in any repeat: {result['tie_break_applied']}")
    print("\nStrategy summary:")
    print(result["summary"].to_string(index=False, formatters={
        "Selection frequency": "{:.3f}".format,
        "Mean inner-CV AP": "{:.3f}".format,
        "Mean clean-test AP": "{:.3f}".format,
        "Mean clean-test ROC-AUC": "{:.3f}".format,
        "Mean clean-test recall": "{:.3f}".format,
        "Mean clean-test precision": "{:.3f}".format,
    }))
    result["summary"].to_csv(args.output, index=False)
    result["runs"].to_csv(args.output.with_name(args.output.stem + "_runs.csv"), index=False)
    print(f"\nSaved summary to: {args.output.resolve()}")
    print(f"Saved repeat-level selection data to: {args.output.with_name(args.output.stem + '_runs.csv').resolve()}")
    print("\nCAUTION: This is synthetic data only, not a healthcare experiment or a paper result.")


if __name__ == "__main__":
    main()
