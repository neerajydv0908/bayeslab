"""Reference answers and an independent enumeration oracle for exact inference."""

from copy import deepcopy
from fractions import Fraction
from itertools import product
import math
import random
import unittest

from bayesnet import DecisionAnalysis, Enumeration, Inference, ImpossibleEvidence, ModelError, Network
from bayesnet.examples import load_example


def enumerate_probability(network, event, evidence=None, action=None):
    """Independent full-joint enumeration, intentionally not using Factor."""
    evidence = evidence or {}
    numerator = denominator = 0.0
    for values in product(*(network.domains[name] for name in network.chance_names)):
        assignment = dict(zip(network.chance_names, values))
        if network.decision:
            assignment[network.decision.name] = action
        probability = 1.0
        for node in network.chance.values():
            probability *= node.cpt[tuple(assignment[parent] for parent in node.parents) + (assignment[node.name],)]
        if all(assignment[name] == state for name, state in evidence.items()):
            denominator += probability
            if all(assignment[name] == state for name, state in event.items()):
                numerator += probability
    return numerator / denominator


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.alarm = Network(load_example("alarm"))
        self.engine = Inference(self.alarm)

    def test_prior_marginal(self):
        self.assertAlmostEqual(self.engine.probability({"Burglary": "yes"}), .001, places=12)
        self.assertAlmostEqual(self.engine.probability({"Alarm": "yes"}), .002516442, places=12)

    def test_joint_assignment(self):
        event = {"Burglary":"yes", "Earthquake":"no", "Alarm":"yes", "JohnCalls":"yes", "MaryCalls":"yes"}
        self.assertAlmostEqual(self.engine.probability(event), .001*.998*.94*.9*.7, places=12)

    def test_burglary_given_both_calls(self):
        self.assertAlmostEqual(self.engine.probability({"Burglary":"yes"}, {"JohnCalls":"yes", "MaryCalls":"yes"}), .2841718353643929, places=12)

    def test_sprinkler_posterior(self):
        engine = Inference(Network(load_example("sprinkler")))
        self.assertAlmostEqual(engine.probability({"Rain":"yes"}, {"WetGrass":"yes"}), .7079276773296245, places=12)

    def test_distribution_normalized_and_observed_target(self):
        distribution = self.engine.distribution("Burglary", {"JohnCalls":"yes"})
        self.assertAlmostEqual(sum(distribution.values()), 1)
        self.assertEqual(self.engine.distribution("Burglary", {"Burglary":"no"}), {"yes":0, "no":1})

    def test_conflicting_events_are_zero(self):
        self.assertEqual(self.engine.probability({"Burglary":"yes"}, {"Burglary":"no"}), 0)
        self.assertEqual(self.engine.probability({"Burglary":"yes", "Alarm":"yes"}, {"Burglary":"no"}), 0)

    def test_impossible_evidence_is_undefined(self):
        data = load_example("sprinkler")
        engine = Inference(Network(data))
        with self.assertRaises(ImpossibleEvidence):
            engine.probability({"Cloudy":"yes"}, {"Sprinkler":"no", "Rain":"no", "WetGrass":"yes"})
        with self.assertRaises(ImpossibleEvidence):
            engine.probability({"Cloudy":"yes", "Rain":"yes"}, {"Sprinkler":"no", "Rain":"no", "WetGrass":"yes"})

    def test_invalid_query_inputs(self):
        for event, evidence in [({}, {}), ({"Missing":"yes"}, {}), ({"Rain":"maybe"}, {}), ({"Alarm":"yes"}, {"Alarm":"maybe"}), ([], {})]:
            with self.subTest(event=event, evidence=evidence), self.assertRaises(ModelError):
                self.engine.probability(event, evidence)
        with self.assertRaises(ModelError):
            self.engine.probability({"Alarm":"yes"}, action="take")

    def test_umbrella_does_not_change_weather(self):
        engine = Inference(Network(load_example("umbrella")))
        self.assertAlmostEqual(engine.probability({"Rain":"yes"}), .3)
        self.assertAlmostEqual(engine.probability({"Rain":"yes"}, {"Forecast":"rainy"}, action="take"), 12/19)

    def test_queries_match_independent_enumeration(self):
        rng = random.Random(42)
        for name in ("alarm", "umbrella", "sprinkler"):
            network = Network(load_example(name))
            action = network.decision.states[0] if network.decision else None
            engine = Inference(network)
            for _ in range(45):
                target_names = rng.sample(network.chance_names, rng.randint(1, len(network.chance_names)))
                available = [n for n in network.chance_names if n not in target_names]
                evidence_names = rng.sample(available, rng.randint(0, len(available)))
                event = {n:rng.choice(network.domains[n]) for n in target_names}
                evidence = {n:rng.choice(network.domains[n]) for n in evidence_names}
                with self.subTest(name=name, event=event, evidence=evidence):
                    try:
                        reference = enumerate_probability(network, event, evidence, action)
                    except ZeroDivisionError:
                        with self.assertRaises(ImpossibleEvidence):
                            engine.probability(event, evidence, action)
                    else:
                        self.assertAlmostEqual(engine.probability(event, evidence, action), reference, places=11)

    def test_multistate_disconnected_network(self):
        data = {"chance_nodes":[
            {"name":"Weather", "states":["sun", "rain", "snow"], "cpt":[{"when":{}, "probabilities":{"sun":.5,"rain":.3,"snow":.2}}]},
            {"name":"Coin", "states":["heads", "tails"], "cpt":[{"when":{}, "probabilities":{"heads":.6,"tails":.4}}]},
        ]}
        engine = Inference(Network(data))
        self.assertAlmostEqual(engine.probability({"Weather":"snow"}, {"Coin":"heads"}), .2)
        self.assertAlmostEqual(engine.probability({"Weather":"rain", "Coin":"tails"}), .12)

    def test_irrelevant_variables_are_pruned(self):
        # Z is isolated. Eliminating the A/C cluster would need a 262,144-entry
        # factor, so queries that do not involve the cluster must skip it.
        states = [f"v{i}" for i in range(8)]
        def uniform(name, parents=()):
            return {"name": name, "states": states, "parents": list(parents), "cpt": [
                {"when": dict(zip(parents, combo)), "probabilities": {s: 1 / 8 for s in states}}
                for combo in product(states, repeat=len(parents))]}
        roots = [f"A{i}" for i in range(6)]
        families = [(0, 1, 2, 3), (2, 3, 4, 5), (0, 1, 4, 5), (0, 1, 2, 4), (0, 1, 3, 5)]
        data = {"chance_nodes": [
            {"name": "Z", "states": ["yes", "no"], "cpt": [{"when": {}, "probabilities": {"yes": .3, "no": .7}}]},
            *(uniform(name) for name in roots),
            *(uniform(f"C{i}", [roots[j] for j in family]) for i, family in enumerate(families))]}
        engine = Inference(Network(data))
        self.assertAlmostEqual(engine.probability({"Z": "yes"}), .3, places=15)
        self.assertAlmostEqual(engine.probability({"Z": "yes"}, {"C0": "v1"}), .3, places=15)
        self.assertAlmostEqual(engine.probability({"C0": "v0"}, {"C1": "v1"}), 1 / 8, places=15)
        # A query that genuinely needs the whole cluster still hits the size cap.
        with self.assertRaisesRegex(ModelError, "200,000"):
            engine.probability({"A0": "v0"}, {f"C{i}": "v0" for i in range(5)})

    def test_tiny_evidence_probabilities_do_not_underflow(self):
        # Sixteen sensors, each "on" with probability around 1e-25, make the evidence
        # about 1e-391 likely: far below the smallest float, about 2.2e-308.
        likelihood = {"a": 1e-25, "b": 3e-25}
        sensors = [f"S{i}" for i in range(16)]
        nodes = [{"name": "H", "states": ["a", "b"], "cpt": [{"when": {}, "probabilities": {"a": .4, "b": .6}}]}]
        nodes += [{"name": name, "states": ["on", "off"], "parents": ["H"], "cpt": [
            {"when": {"H": h}, "probabilities": {"on": p, "off": 1 - p}} for h, p in likelihood.items()]}
            for name in sensors]
        utility = {("a", "go"): 10, ("b", "go"): -1, ("a", "stay"): 0, ("b", "stay"): 0}
        network = Network({"chance_nodes": nodes,
                           "decision_nodes": [{"name": "D", "states": ["go", "stay"], "information": sensors}],
                           "utility_nodes": [{"name": "U", "parents": ["H", "D"], "table": [
                               {"when": {"H": h, "D": d}, "value": v} for (h, d), v in utility.items()]}]})
        evidence = {name: "on" for name in sensors}
        # Exact answers from rational arithmetic on the same binary values.
        mass = {h: Fraction(prior) * Fraction(likelihood[h]) ** 16 for h, prior in (("a", .4), ("b", .6))}
        total = mass["a"] + mass["b"]
        posterior = mass["a"] / total
        log10_total = math.log10(total.numerator) - math.log10(total.denominator)
        self.assertLess(log10_total, -390)
        for engine in (Inference(network), Enumeration(network)):
            with self.subTest(engine=type(engine).__name__):
                result = engine.query({"H": "a"}, evidence)
                self.assertAlmostEqual(result["probability"] / float(posterior), 1, places=12)
                self.assertAlmostEqual(result["log10_evidence_probability"], log10_total, places=9)
                self.assertEqual(result["evidence_probability"], 0.0)  # the float itself underflows
                self.assertTrue(any(step["operation"] == "rescale" for step in result["trace"]))
        go = float(posterior * 10 - (1 - posterior))
        for method in ("elimination", "enumeration"):
            with self.subTest(method=method):
                result = DecisionAnalysis(network, method).analyze(evidence)
                self.assertAlmostEqual(result["actions"][0]["expected_utility"], go, places=12)
                self.assertEqual(result["best_action"], "stay")

    def test_trace_reports_pruned_variables(self):
        engine = Inference(Network(load_example("umbrella")))
        trace = engine.query({"Rain": "yes"}, {"Forecast": "rainy"})["trace"]
        self.assertEqual([step["variables"] for step in trace if step["operation"] == "prune"], [["WetRoad"]])
        trace = engine.query({"Rain": "yes"}, {"Forecast": "rainy", "WetRoad": "yes"})["trace"]
        self.assertFalse(any(step["operation"] == "prune" for step in trace))


class ValidationTests(unittest.TestCase):
    def test_rejects_cycles(self):
        data = load_example("umbrella")
        data["decision_nodes"][0]["information"] = ["Forecast"]
        data["chance_nodes"][0]["parents"] = ["Umbrella"]
        data["chance_nodes"][0]["cpt"] = [
            {"when":{"Umbrella":action}, "probabilities":{"yes":.3, "no":.7}} for action in ("take", "leave")]
        with self.assertRaisesRegex(ModelError, "cycle"):
            Network(data)

    def test_rejects_invalid_cpts(self):
        bad_values = [{"yes":.2,"no":.2}, {"yes":-.1,"no":1.1}, {"yes":float('nan'),"no":.5}, {"yes":True,"no":0}, {"yes":1}, {"yes":10**1000,"no":0}]
        for bad in bad_values:
            with self.subTest(bad=str(bad)[:80]), self.assertRaises(ModelError):
                data = load_example("alarm")
                data["chance_nodes"][0]["cpt"][0]["probabilities"] = bad
                Network(data)

    def test_rejects_missing_and_duplicate_rows(self):
        data = load_example("umbrella")
        data["chance_nodes"][1]["cpt"].pop()
        with self.assertRaises(ModelError): Network(data)
        data = load_example("umbrella")
        data["chance_nodes"][1]["cpt"][1] = deepcopy(data["chance_nodes"][1]["cpt"][0])
        with self.assertRaisesRegex(ModelError, "Duplicate"): Network(data)

    def test_rejects_duplicate_names_unknown_parents_and_information(self):
        for mutation in ("duplicate", "parent", "information"):
            data = load_example("umbrella")
            if mutation == "duplicate": data["decision_nodes"][0]["name"] = "Rain"
            elif mutation == "parent": data["chance_nodes"][1]["parents"] = ["Missing"]
            else: data["decision_nodes"][0]["information"] = ["Missing"]
            with self.subTest(mutation=mutation), self.assertRaises(ModelError): Network(data)

    def test_supported_scope_is_explicit(self):
        data = load_example("umbrella")
        data["decision_nodes"].append({"name":"Other", "states":["a","b"], "information":[]})
        with self.assertRaisesRegex(ModelError, "one decision"): Network(data)

    def test_cpt_errors_name_the_row_and_the_problem(self):
        data = load_example("alarm")
        row = data["chance_nodes"][2]["cpt"][1]  # Alarm given Burglary=yes, Earthquake=no
        for probabilities, message in (
                ({"yes": .94, "no": .05}, r"Alarm given Burglary=yes, Earthquake=no add up to 0\.99"),
                ({"yes": 1.2, "no": -.2}, r"P\(Alarm=yes \| Burglary=yes, Earthquake=no\) is 1\.2"),
                ({"yes": 1}, r"must list exactly the states yes, no \(missing no\)"),
                ({"yes": .5, "no": .5, "maybe": 0}, r"unknown maybe")):
            row["probabilities"] = probabilities
            with self.subTest(probabilities=probabilities), self.assertRaisesRegex(ModelError, message):
                Network(data)
        data = load_example("alarm")
        data["chance_nodes"][0]["cpt"][0]["probabilities"] = {"yes": .2, "no": .7}
        with self.assertRaisesRegex(ModelError, r"The probabilities of Burglary add up to 0\.9,"):
            Network(data)

    def test_rejects_utilities_too_large_to_add_up(self):
        data = load_example("umbrella")
        data["utility_nodes"].append({"name": "Extra", "parents": [], "table": [{"when": {}, "value": 1e308}]})
        with self.assertRaisesRegex(ModelError, "too large"):
            Network(data)

    def test_observation_errors_say_what_is_wrong(self):
        engine = Inference(Network(load_example("umbrella")))
        for evidence, message in (({"Temperature": "hot"}, "not a chance variable"),
                                  ({"Umbrella": "take"}, "is the decision"),
                                  ({"Forecast": "sunny"}, r"'sunny' is not a state of Forecast \(states: rainy, dry\)")):
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ModelError, message):
                engine.probability({"Rain": "yes"}, evidence)


if __name__ == "__main__":
    unittest.main()
