"""Bundled networks, also available as editable JSON files."""

import json
from pathlib import Path

EXAMPLE_DIR = Path(__file__).with_name("examples")
EXAMPLES = [
    {"id": "umbrella", "name": "The umbrella decision", "category": "Decision network",
     "description": "Use a weather forecast to decide whether to bring an umbrella."},
    {"id": "alarm", "name": "The burglar alarm", "category": "Bayesian network",
     "description": "Infer a burglary from an alarm and calls from two neighbors."},
    {"id": "sprinkler", "name": "Why is the grass wet?", "category": "Bayesian network",
     "description": "Explore how rain and a sprinkler explain wet grass."},
]


def load_example(example_id):
    if example_id not in {example["id"] for example in EXAMPLES}:
        raise ValueError("Unknown example. Choose umbrella, alarm, or sprinkler.")
    return json.loads((EXAMPLE_DIR / f"{example_id}.json").read_text(encoding="utf-8"))

