
"""Local administration without credential management."""

import argparse

from rag_app.config import Settings
from rag_app.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["maintenance"],
        help="Administrative command to run.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Retention window in days for expired conversations and cached responses.",
    )
    args = parser.parse_args()

    if args.command == "maintenance":
        settings = Settings.from_env()
        store = Store(settings.database_path)
        store.maintenance(args.days)

        print(
            f"Expired conversations and cached responses older than "
            f"{args.days} days were removed."
        )


if __name__ == "__main__":
    main()