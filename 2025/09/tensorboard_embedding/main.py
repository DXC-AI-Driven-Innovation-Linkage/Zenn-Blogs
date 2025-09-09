"""tf-embed: small entrypoint

This module provides a tiny CLI entrypoint. The repository's primary demo is the
`test.ipynb` notebook. Running this script prints a short message and can be
extended later to programmatically run parts of the workflow.
"""

from __future__ import annotations


def main() -> int:
    """Prints a short informational message and returns an exit code.

    Returns 0 on success.
    """
    print("Hello from tf-embed! The main demo is in 'test.ipynb'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
