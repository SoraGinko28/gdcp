"""Standalone independent checkpoint reader; deliberately imports no torch/GDCP."""

import argparse
import json
import sys
from pathlib import Path

from tests.support.gdcp_reader import independent_reader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("oracle")
    args = parser.parse_args()
    result, _counts, _manifest = independent_reader(args.checkpoint)
    expected = json.loads(Path(args.oracle).read_text())
    actual = result
    assert actual == expected
    assert "torch" not in sys.modules
    assert "gdcp" not in sys.modules and "gdtensor" not in sys.modules
    print(
        "ACCEPTANCE 5 PASS independent reader imports only stdlib/safetensors and every tensor matches oracle",
        flush=True,
    )


if __name__ == "__main__":
    main()
