"""CLI entry point for ANN training.

Usage:
    python -m models.photoz.ANN.train --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .core import Config, Trainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a regression MLP for photo-z."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args(argv)

    config = Config()
    if Path(args.config).exists():
        config.update_from_yaml(args.config)
    else:
        logging.warning(
            f"Config file not found at {args.config!r}. Using defaults."
        )
    config.config_path = args.config

    Trainer(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
