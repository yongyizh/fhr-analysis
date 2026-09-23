"""Train FUNet.

    funet-train [fetal-config.yaml] [--diagnostics]

Everything below the config lives in common: see common.phases.train.run_training.
"""

import argparse
import os
import sys

from common.config import load_config
from common.phases.train import BEST_MODEL, run_training

from funet.task import FUNetTask

DIAGNOSTICS_PLOT = "snippet_diagnostics.png"


def parse_args(argv):
    p = argparse.ArgumentParser(prog="funet-train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default="fetal-config.yaml",
                   help="config to train (default: fetal-config.yaml)")
    # --- diagnostics (optional; delete this argument and the block in main to remove) ---
    p.add_argument("--diagnostics", action="store_true",
                   help="after training, plot the best model against a few validation "
                        "snippets: its activity against the target, and the BPM traces those "
                        "produce. Writes " + DIAGNOSTICS_PLOT + " next to the checkpoints.")
    # --- end diagnostics ---
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    task = FUNetTask()
    config = load_config(args.config, task)
    print(f"Loaded config: '{args.config}'")
    run_training(task, config)

    # --- diagnostics (optional; delete this block and the CLI argument to remove) ---
    if args.diagnostics and not config.data.val_dir:
        print("--diagnostics skipped: it plots validation snippets and there is no val_dir")
    elif args.diagnostics:
        from common.diagnostics import plot_snippet_diagnostics
        plot_snippet_diagnostics(
            task, config,
            checkpoint_path=os.path.join(config.model_dir, BEST_MODEL),
            out_path=os.path.join(config.model_dir, DIAGNOSTICS_PLOT),
        )
    # --- end diagnostics ---


if __name__ == "__main__":
    main()
