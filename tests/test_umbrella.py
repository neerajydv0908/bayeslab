"""Acceptance tests for the supplied teaching model and new reusable APIs."""

from copy import deepcopy
import importlib.util
from itertools import product
import json
import math
import unittest

from bayesnet import DecisionAnalysis, Enumeration, ImpossibleEvidence, Inference, ModelError, Network
from bayesnet.examples import load_example
from bayesnet.inference import Factor
from experiments.compare_inference import chain
from tests.test_server import request


class UmbrellaTests(unittest.TestCase):
    def setUp(self):
        self.network = Network(load_example("umbrella"))

    def test_all_eight_worlds_match_hand_calculations(self):
        expected = [.216, .024, .054, .006, .014, .126, .056, .504]
        rows = list(Enumeration(self.network).worlds())
        self.assertEqual(len(rows), 8)
        self.assertAlmostEqual(math.fsum(row["probability"] for row in rows), 1)
        for row, value in zip(rows, expected):
            for engine in (Enumeration(self.network), Inference(self.network)):
                self.assertAlmostEqual(engine.joint_probability(row["assignment"]), value, places=12)

    def test_probability_acceptance_values_and_hidden_variable(self):
        for engine in (Enumeration(self.network), Inference(self.network)):
            self.assertAlmostEqual(engine.event_probability({"Forecast": "rainy"}), .38)
            self.assertAlmostEqual(engine.event_probability({"WetRoad": "yes"}), .34)
            self.assertAlmostEqual(engine.event_probability({"Rain": "yes", "Forecast": "rainy"}), .24)
            self.assertAlmostEqual(engine.conditional_probability({"Rain": "yes"}, {"Forecast": "rainy"}), 12/19)
            self.assertAlmostEqual(engine.conditional_probability({"Rain": "yes"}, {"Forecast": "dry"}), 3/31)
            self.assertAlmostEqual(engine.event_probability({}), 1)
            with self.assertRaisesRegex(ModelError, "every chance"):
                engine.joint_probability({"Rain": "yes"})

    def test_enumeration_and_elimination_agree_on_all_partial_assignments(self):
        enum, ve = Enumeration(self.network), Inference(self.network)
        for choices in product(range(5), repeat=3):
            event, evidence = {}, {}
            for n, c in zip(self.network.chance_names, choices):
                if c in (1, 2): event[n] = self.network.domains[n][c-1]
                if c in (3, 4): evidence[n] = self.network.domains[n][c-3]
            if event:
                with self.subTest(event=event, evidence=evidence):
                    self.assertAlmostEqual(enum.probability(event, evidence), ve.probability(event, evidence), places=12)

    def test_enumeration_invalid_and_impossible_inputs(self):
        engine = Enumeration(self.network)
        for event, evidence in [({}, {}), ({"Rain": "Maybe"}, {}), ({"Temperature": "Hot"}, {}), ({"Rain": "yes"}, []), ([], {})]:
            with self.subTest(event=event), self.assertRaises(ModelError):
                engine.probability(event, evidence)
        self.assertEqual(engine.probability({"Rain": "yes"}, {"Rain": "no"}), 0)
        self.assertEqual(engine.distribution("Rain", {"Rain": "yes"}), {"yes": 1, "no": 0})
        data = load_example("umbrella")
        data["chance_nodes"][0]["cpt"][0]["probabilities"] = {"yes": 0, "no": 1}
        network = Network(data)
        for engine in (Enumeration(network), Inference(network)):
            with self.assertRaises(ImpossibleEvidence):
                engine.probability({"Rain": "no"}, {"Rain": "yes"})
        with self.assertRaisesRegex(ModelError, "Enumeration exceeds"):
            list(Enumeration(chain(21)).worlds())

    def test_enumeration_visits_only_consistent_nonzero_worlds(self):
        trace = Enumeration(self.network).query({"Rain": "yes"}, {"Forecast": "rainy"})["trace"]
        self.assertIn("through the 4 worlds", trace[0]["description"])  # Rain x WetRoad, Forecast fixed
        self.assertIn("of 8 possible", trace[0]["description"])
        # Wet grass is impossible with neither rain nor sprinkler, so those branches are skipped.
        sprinkler = Enumeration(Network(load_example("sprinkler")))
        result = sprinkler.query({"Rain": "yes"}, {"WetGrass": "yes"})
        self.assertIn("through the 6 worlds", result["trace"][0]["description"])
        self.assertAlmostEqual(result["probability"], .7079276773296245, places=12)

    def test_expected_utility_acceptance_values_for_both_engines(self):
        for method in ("enumeration", "elimination"):
            analysis = DecisionAnalysis(self.network, method)
            for forecast, take, leave, best in (("rainy", 82/19, -205/19, "take"), ("dry", -32/31, 80/31, "leave")):
                result = analysis.analyze({"Forecast": forecast})
                self.assertAlmostEqual(result["actions"][0]["expected_utility"], take)
                self.assertAlmostEqual(result["actions"][1]["expected_utility"], leave)
                self.assertEqual(result["best_action"], best)
            with self.assertRaises(ModelError):
                analysis.analyze({"WetRoad": "yes"})

    def test_threshold_tie_and_changed_preferences(self):
        result = DecisionAnalysis(self.network).sensitivity("Rain", {"Forecast": "rainy"})
        self.assertAlmostEqual(result["threshold"], .2)
        self.assertAlmostEqual(result["posterior"], 12/19)
        self.assertEqual(result["points"][20]["actions"], {"take": 0, "leave": 0})
        data = deepcopy(self.network.source)
        data["chance_nodes"][0]["cpt"][0]["probabilities"] = {"yes": .2, "no": .8}
        self.assertEqual(DecisionAnalysis(Network(data)).analyze()["optimal_actions"], ["take", "leave"])
        data["utility_nodes"][0]["table"][1]["value"] = -40
        self.assertAlmostEqual(DecisionAnalysis(Network(data)).sensitivity("Rain")["threshold"], 7/55)
        for row in data["utility_nodes"][0]["table"]: row["value"] = 3
        self.assertTrue(DecisionAnalysis(Network(data)).sensitivity("Rain")["always_tied"])
        with self.assertRaises(ModelError):
            DecisionAnalysis(self.network).sensitivity("WetRoad")

    def test_factor_operations_keep_correct_scopes_and_mass(self):
        a = Factor(("A",), {("yes",): .3, ("no",): .7})
        b = Factor(("A", "B"), {("yes", "yes"): .8, ("yes", "no"): .2,
                                  ("no", "yes"): .2, ("no", "no"): .8})
        joint = a.multiply(b, {"A": ("yes", "no"), "B": ("yes", "no")})
        self.assertAlmostEqual(joint.sum_out("A").values[("yes",)], .38)
        posterior = joint.restrict({"B": "yes"}).normalize()
        self.assertEqual(posterior.variables, ("A",))
        self.assertAlmostEqual(posterior.values[("yes",)], 12/19)
        with self.assertRaises(ImpossibleEvidence): Factor((), {(): 0}).normalize()

    def test_action_dependent_queries_require_an_explicit_action(self):
        data = {"chance_nodes": [{"name": "Success", "states": ["yes", "no"], "parents": ["Choice"],
                "cpt": [{"when": {"Choice": "act"}, "probabilities": {"yes": .8, "no": .2}},
                        {"when": {"Choice": "wait"}, "probabilities": {"yes": .2, "no": .8}}]}],
                "decision_nodes": [{"name": "Choice", "states": ["act", "wait"]}],
                "utility_nodes": [{"name": "Benefit", "parents": ["Success"], "table": [
                    {"when": {"Success": "yes"}, "value": 10}, {"when": {"Success": "no"}, "value": -5}]}]}
        network = Network(data)
        for engine in (Enumeration(network), Inference(network)):
            with self.assertRaises(ModelError): engine.probability({"Success": "yes"})
            self.assertAlmostEqual(engine.probability({"Success": "yes"}, action="act"), .8)
            self.assertAlmostEqual(engine.probability({"Success": "yes"}, action="wait"), .2)
        for method in ("enumeration", "elimination"):
            result = DecisionAnalysis(network, method).analyze()
            self.assertAlmostEqual(result["maximum_expected_utility"], 7)

    def test_new_http_routes_and_method_validation(self):
        base = {"model": self.network.source}
        status, _, body = request("/api/worlds", base)
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["worlds"]), 8)
        status, _, body = request("/api/sensitivity", {**base, "target": "Rain"})
        self.assertEqual(status, 200)
        self.assertAlmostEqual(json.loads(body)["threshold"], .2)
        status, _, body = request("/api/query", {**base, "method": "enumeration", "event": {"WetRoad": "yes"}})
        self.assertEqual(status, 200)
        self.assertAlmostEqual(json.loads(body)["probability"], .34)
        self.assertEqual(request("/api/query", {**base, "method": "unknown", "event": {"Rain": "yes"}})[0], 400)


@unittest.skipUnless(importlib.util.find_spec("pgmpy"), "Optional pgmpy reference dependency is not installed")
class PgmpyTests(unittest.TestCase):
    def test_external_reference_matches_both_algorithms(self):
        from experiments.compare_inference import compare
        report = compare(include_pgmpy=True, repeats=1, sizes=(4,))
        self.assertEqual(report["probability_queries_checked"], 278)
        self.assertLess(report["maximum_absolute_difference"], 1e-10)

    def test_reference_handles_overlapping_and_impossible_evidence(self):
        from bayesnet.reference import PgmpyReference
        data = load_example("umbrella")
        reference = PgmpyReference(Network(data))
        self.assertEqual(reference.probability({"Rain": "yes"}, {"Rain": "no"}), 0)
        self.assertEqual(reference.probability({"Rain": "yes"}, {"Rain": "yes"}), 1)
        data["chance_nodes"][0]["cpt"][0]["probabilities"] = {"yes": 0, "no": 1}
        with self.assertRaises(ImpossibleEvidence):
            PgmpyReference(Network(data)).probability({"Rain": "no"}, {"Rain": "yes"})
