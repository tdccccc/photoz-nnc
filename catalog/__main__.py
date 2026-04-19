"""Unified CLI for the catalog pipeline.

Usage::

    python -m catalog --stage lsdr10
    python -m catalog --stage ps1dr2
    python -m catalog --stage all
    python -m catalog --stage merge
    python -m catalog --stage publish
    python -m catalog --stage check

    # Override the config file (default: catalog/config.yaml or
    # $PHOTOZ_CATALOG_CONFIG):
    python -m catalog --config /path/to/config.yaml --stage merge
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Photo-z catalog pipeline (predict / merge / publish)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "    python -m catalog --stage lsdr10\n"
            "    python -m catalog --stage all\n"
            "    python -m catalog --stage merge\n"
            "    python -m catalog --stage publish\n"
            "    python -m catalog --stage check\n"
        ),
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["lsdr10", "ps1dr2", "all", "merge", "publish", "check"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to config.yaml. Falls back to "
            "$PHOTOZ_CATALOG_CONFIG, then catalog/config.yaml."
        ),
    )
    args = parser.parse_args()

    # Set the env var early so that common.load_config() picks it up
    # on its first import inside sub-modules.
    if args.config is not None:
        os.environ["PHOTOZ_CATALOG_CONFIG"] = args.config

    if args.stage in ("lsdr10", "ps1dr2", "all"):
        from .predict import run_lsdr10, run_ps1dr2

        if args.stage in ("lsdr10", "all"):
            run_lsdr10()
        if args.stage in ("ps1dr2", "all"):
            if args.stage == "all":
                print("\n" + "=" * 60 + "\n")
            run_ps1dr2()

    elif args.stage == "merge":
        from .merge import merge_surveys

        merge_surveys()

    elif args.stage == "publish":
        from .publish import publish_catalog

        publish_catalog()

    elif args.stage == "check":
        from .publish import check_published_catalog

        check_published_catalog()


if __name__ == "__main__":
    main()
