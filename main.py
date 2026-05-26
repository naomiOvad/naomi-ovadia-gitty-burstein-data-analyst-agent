"""Entry point: `python main.py` drops into the interactive agent CLI.

Optional `--session <id>` argument is reserved for Task 2 (persistent
conversation memory) and has no effect yet.
"""

import argparse

from src.cli import run_cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent (Bitext dataset)."
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session ID for persistent memory (used in Task 2).",
    )
    args = parser.parse_args()
    run_cli(session_id=args.session)


if __name__ == "__main__":
    main()
