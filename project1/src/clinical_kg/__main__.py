"""Allow `python3 -m clinical_kg <command>` without installing the package."""

from clinical_kg.cli import main

if __name__ == "__main__":
    main()
