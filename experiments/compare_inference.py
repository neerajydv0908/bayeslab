"""Run with python -m experiments.compare_inference --output docs/benchmark.json."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
from itertools import product
import json
import math
from pathlib import Path
import platform
import random
from statistics import median
from time import perf_counter

from bayesnet import DecisionAnalysis, Enumeration, ImpossibleEvidence, Inference, Network
from bayesnet.examples import load_example
from bayesnet.reference import PgmpyReference


def chain(size):
    """Binary chain with small factors; intentionally favorable to elimination."""
    nodes = []
    for i in range(size):
        parent = f"X{i-1}"
        nodes.append({"name": f"X{i}", "states": ["yes", "no"],
                      "parents": [parent] if i else [],
                      "cpt": [{"when": {parent: s}, "probabilities": {"yes": p, "no": 1-p}}
                              for s, p in (("yes", .8), ("no", .2))] if i else
                             [{"when": {}, "probabilities": {"yes": .3, "no": .7}}]})
    return Network({"name": f"Binary chain ({size})", "chance_nodes": nodes})


def assert_equal(values, label):
    if not all(math.isclose(values[0], v, rel_tol=0, abs_tol=1e-10) for v in values[1:]):
        raise AssertionError(f"Inference mismatch for {label}: {values}")


def compare(include_pgmpy=True, repeats=5, sizes=(4, 8, 12, 16)):
    checks = 0
    max_error = 0.0
    randomizer = random.Random(2026)
    for example in ("umbrella", "alarm", "sprinkler"):
        network = Network(load_example(example))
        engines = [Enumeration(network), Inference(network)]
        if include_pgmpy:
            engines.append(PgmpyReference(network))
        queries = []
        if example == "umbrella":
            # Every disjoint event/evidence combination, including complete worlds.
            for choices in product(range(5), repeat=len(network.chance_names)):
                event, evidence = {}, {}
                for name, choice in zip(network.chance_names, choices):
                    if choice in (1, 2): event[name] = network.domains[name][choice-1]
                    if choice in (3, 4): evidence[name] = network.domains[name][choice-3]
                if event: queries.append((event, evidence))
        for _ in range(60):
            names = list(network.chance_names)
            randomizer.shuffle(names)
            split = randomizer.randint(1, len(names))
            event = {n: randomizer.choice(network.domains[n]) for n in names[:split]}
            evidence = {n: randomizer.choice(network.domains[n]) for n in names[split:]}
            queries.append((event, evidence))
        for event, evidence in queries:
            values = []
            for engine in engines:
                try:
                    values.append(engine.probability(event, evidence))
                except ImpossibleEvidence:
                    values.append(None)
            if None in values:
                if not all(v is None for v in values):
                    raise AssertionError(f"Impossible-evidence mismatch: {values}")
            else:
                assert_equal(values, (example, event, evidence))
                max_error = max(max_error, max(values) - min(values))
            checks += 1

    network = Network(load_example("umbrella"))
    utility_checks = []
    for forecast in ("rainy", "dry"):
        evidence = {"Forecast": forecast}
        for action in network.decision.states:
            values = [DecisionAnalysis(network, method).expected_utility(action, evidence)["expected_utility"]
                      for method in ("enumeration", "elimination")]
            if include_pgmpy:
                reference = PgmpyReference(network)
                table = network.utilities[0].table
                values.append(math.fsum(reference.probability({"Rain": rain}, evidence) * table[(rain, action)]
                                        for rain in network.domains["Rain"]))
            assert_equal(values, (forecast, action))
            utility_checks.append({"forecast": forecast, "action": action, "expected_utility": values[0]})

    timings = []
    for size in sizes:
        network = chain(size)
        engines = {"enumeration": Enumeration(network), "elimination": Inference(network)}
        if include_pgmpy: engines["pgmpy"] = PgmpyReference(network)
        event, evidence = {f"X{size-1}": "yes"}, {"X0": "yes"}
        values = [engine.probability(event, evidence) for engine in engines.values()]
        assert_equal(values, size)
        timing = {"nodes": size, "worlds": 2 ** size, "probability": values[0], "median_ms": {}}
        for name, engine in engines.items():
            samples = []
            for _ in range(repeats):
                start = perf_counter()
                engine.probability(event, evidence)
                samples.append(1000 * (perf_counter() - start))
            timing["median_ms"][name] = median(samples)
        timings.append(timing)
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.machine(),
            "pgmpy": importlib.metadata.version("pgmpy") if include_pgmpy else None,
            "probability_queries_checked": checks, "maximum_absolute_difference": max_error,
            "utility_checks": utility_checks, "repetitions": repeats, "timings": timings,
            "methodology": "Median of repeated warm queries, excluding imports and model construction. Binary chains have low treewidth and favor elimination; timings are not a general speed guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-pgmpy", action="store_true", help="Compare only the two local engines")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1: parser.error("--repeats must be positive")
    try:
        report = compare(not args.without_pgmpy, args.repeats)
    except ImportError:
        parser.exit(1, "Install requirements-reference.txt, or use --without-pgmpy.\n")
    formatted = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted)
    print(formatted, end="")


if __name__ == "__main__":
    main()
