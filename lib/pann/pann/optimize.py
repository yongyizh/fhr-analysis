"""Optuna hyperparameter search for PANNet.

    pann-optimize [fetal-config.yaml] [--trials N] [--epochs N] [--storage URL] [--seed S]

The search space lives in PANNTask.suggest / PANNTask.searched_fields; the driver (trials,
pruning, study resume, best/latest config+model slots) lives in common.phases.optimize, which is
the only place optuna is imported.

Trials are cheap relative to FUNet's: the frozen backbone means a trial only ever fits a small
head, so the search is dominated by backbone forward passes rather than by anything it varies.
"""

import sys

from common.phases.optimize import main as optimize_main

from pann.task import PANNTask


def main(argv=None) -> None:
    optimize_main(PANNTask(), sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
