import subprocess
import sys
from pathlib import Path

def auto_lint(path: str):
    """Run Black and Ruff on the given path."""
    path = Path(path).resolve()

    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        sys.exit(1)

    print(f"📂 Formatting and linting: {path}")

    # Run Black (formatter)
    subprocess.run(["black", str(path)], check=True)

    # Run Ruff (linter + autofix)
    subprocess.run(["ruff", "check", "--fix", str(path)], check=True)

    print("✅ Formatting and linting complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python auto_lint.py <path-to-file-or-folder>")
        sys.exit(1)

    auto_lint(sys.argv[1])

#python auto_lint.py src/
#python auto_lint.py src/exchange.py