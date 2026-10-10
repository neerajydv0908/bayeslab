from itertools import product
import unittest
from unittest.mock import patch

from bayesnet import DecisionAnalysis, Enumeration, ModelError, Network
from bayesnet.examples import load_example


def observed_chain(size=10, observed=4, action_dependent=False):
    """Binary chain X0 -> ... -> X(size-1); the decision sees X0..X(observed-1)."""
    nodes = []
    for i in range(size):
        parents = [f"X{i-1}"] if i else []
        if action_dependent and i == size - 1:
            parents.append("Act")
        rows = []
        for combo in product(*(("go", "stay") if p == "Act" else ("yes", "no") for p in parents)):
            p_yes = .05 + .8 * (combo[0] == "yes") + .1 * ("go" in combo) if parents else .5
            rows.append({"when": dict(zip(parents, combo)), "probabilities": {"yes": p_yes, "no": 1 - p_yes}})
        nodes.append({"name": f"X{i}", "states": ["yes", "no"], "parents": parents, "cpt": rows})
    last = f"X{size-1}"
    payoff = {("yes", "go"): 10, ("no", "go"): -6, ("yes", "stay"): -3, ("no", "stay"): 2}
    return {"chance_nodes": nodes,
            "decision_nodes": [{"name": "Act", "states": ["go", "stay"], "information": [f"X{i}" for i in range(observed)]}],
            "utility_nodes": [{"name": "Payoff", "parents": [last, "Act"], "table": [
                {"when": {last: x, "Act": a}, "value": v} for (x, a), v in payoff.items()]}]}


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.data = load_example("umbrella")
        self.analysis = DecisionAnalysis(Network(self.data))

    def test_expected_utilities_and_meu(self):
        result = self.analysis.analyze()
        self.assertAlmostEqual(result["actions"][0]["expected_utility"], 1)
        self.assertAlmostEqual(result["actions"][1]["expected_utility"], -2.5)
        self.assertEqual(result["best_action"], "take")
        self.assertAlmostEqual(result["maximum_expected_utility"], 1)

    def test_optimal_contingent_policy(self):
        result = self.analysis.analyze()
        policy = {row["when"]["Forecast"]:row["action"] for row in result["policy"]}
        self.assertEqual(policy, {"rainy":"take", "dry":"leave"})
        self.assertAlmostEqual(result["policy_expected_utility"], 3.24)
        self.assertAlmostEqual(result["value_of_information"], 2.24)
        self.assertAlmostEqual(sum(row["probability"] for row in result["policy"]), 1)

    def test_observed_forecast(self):
        rainy = self.analysis.analyze({"Forecast":"rainy"})
        self.assertAlmostEqual(rainy["maximum_expected_utility"], 4.315789473684211)
        self.assertEqual(rainy["best_action"], "take")
        self.assertAlmostEqual(rainy["value_of_information"], 0)
        dry = self.analysis.analyze({"Forecast":"dry"})
        self.assertEqual(dry["best_action"], "leave")
        self.assertAlmostEqual(dry["maximum_expected_utility"], 2.5806451612903225)

    def test_non_available_observations_rejected(self):
        with self.assertRaises(ModelError): self.analysis.analyze({"Rain":"yes"})
        with self.assertRaises(ModelError): self.analysis.expected_utility("invalid")

    def test_evidence_not_observed_before_deciding_is_explained(self):
        with self.assertRaisesRegex(ModelError, "WetRoad is not observed before the Umbrella decision.*Forecast"):
            self.analysis.analyze({"WetRoad": "yes"})

    def test_worthless_information_is_worth_exactly_zero(self):
        # A forecast that says "rainy" equally often whatever the weather carries no
        # information. Rounding used to report its value as about 1e-15 instead of 0.
        for p_rain, p_rainy in ((.1, .35), (.4, .1), (.4, .2), (.7, .35)):
            data = load_example("umbrella")
            data["chance_nodes"][0]["cpt"][0]["probabilities"] = {"yes": p_rain, "no": round(1 - p_rain, 10)}
            for row in data["chance_nodes"][1]["cpt"]:
                row["probabilities"] = {"rainy": p_rainy, "dry": round(1 - p_rainy, 10)}
            for method in ("elimination", "enumeration"):
                with self.subTest(p_rain=p_rain, p_rainy=p_rainy, method=method):
                    self.assertEqual(DecisionAnalysis(Network(data), method).analyze()["value_of_information"], 0.0)

    def test_additive_utility_and_all_tied_actions(self):
        self.data["utility_nodes"][0]["table"] = [{**row,"value":-5} for row in self.data["utility_nodes"][0]["table"]]
        self.data["utility_nodes"].append({"name":"Constant", "parents":[], "table":[{"when":{},"value":-10}]})
        result = DecisionAnalysis(Network(self.data)).analyze()
        self.assertAlmostEqual(result["maximum_expected_utility"], -15)
        self.assertEqual(result["optimal_actions"], ["take", "leave"])
        self.assertAlmostEqual(result["value_of_information"], 0)

    def test_zero_probability_policy_branch(self):
        for row in self.data["chance_nodes"][1]["cpt"]:
            row["probabilities"] = {"rainy":1,"dry":0}
        result = DecisionAnalysis(Network(self.data)).analyze()
        self.assertIsNone(result["policy"][1]["action"])
        self.assertEqual(result["policy"][1]["probability"], 0)
        self.assertAlmostEqual(result["policy_expected_utility"], 1)

    def test_action_changes_downstream_outcomes(self):
        # Independent reference: act = .8*100 - 30 = 50; wait = .2*100 = 20.
        data = {"chance_nodes":[{"name":"Success", "states":["yes","no"], "parents":["Choice"], "cpt":[
            {"when":{"Choice":"act"}, "probabilities":{"yes":.8,"no":.2}},
            {"when":{"Choice":"wait"}, "probabilities":{"yes":.2,"no":.8}}]}],
            "decision_nodes":[{"name":"Choice", "states":["act","wait"], "information":[]}],
            "utility_nodes":[{"name":"Benefit", "parents":["Success"], "table":[
                {"when":{"Success":"yes"},"value":100}, {"when":{"Success":"no"},"value":0}]},
                {"name":"Cost", "parents":["Choice"], "table":[
                {"when":{"Choice":"act"},"value":-30}, {"when":{"Choice":"wait"},"value":0}]}]}
        result = DecisionAnalysis(Network(data)).analyze()
        self.assertAlmostEqual(result["actions"][0]["expected_utility"], 50)
        self.assertAlmostEqual(result["actions"][1]["expected_utility"], 20)
        self.assertEqual(result["best_action"], "act")

    def test_enumeration_policy_visits_each_world_once(self):
        # 1,024 worlds and 16 policy rows. Re-enumerating for every branch and
        # action would cost (2 + 1 + 16 * 2) * 1,024 = 35,840 world visits.
        walk = Enumeration._walk
        for action_dependent, visits in ((False, 1024), (True, 2048)):
            network = Network(observed_chain(action_dependent=action_dependent))
            seen = []

            def counting_walk(engine, evidence, action, leaf):
                def counted(world, value, exponent):
                    seen.append(1)
                    leaf(world, value, exponent)
                walk(engine, evidence, action, counted)

            with patch.object(Enumeration, "_walk", counting_walk):
                fast = DecisionAnalysis(network, "enumeration").analyze()
            reference = DecisionAnalysis(network, "elimination").analyze()
            with self.subTest(action_dependent=action_dependent):
                self.assertEqual(len(seen), visits)  # each world once per action
                self.assertEqual(fast["optimal_actions"], reference["optimal_actions"])
                for key in ("maximum_expected_utility", "policy_expected_utility", "value_of_information"):
                    self.assertAlmostEqual(fast[key], reference[key], places=12)
                for row, expected in zip(fast["actions"], reference["actions"]):
                    self.assertAlmostEqual(row["expected_utility"], expected["expected_utility"], places=12)
                self.assertEqual(len(fast["policy"]), 16)
                self.assertEqual({row["action"] for row in fast["policy"]}, {"go", "stay"})
                for row, expected in zip(fast["policy"], reference["policy"]):
                    self.assertEqual((row["when"], row["optimal_actions"]), (expected["when"], expected["optimal_actions"]))
                    self.assertAlmostEqual(row["probability"], expected["probability"], places=12)
                    self.assertAlmostEqual(row["expected_utility"], expected["expected_utility"], places=12)

    def test_ties_are_relative_to_the_utility_scale(self):
        tie = {("yes", "take"): 7, ("no", "take"): -12, ("yes", "leave"): 0, ("no", "leave"): 0}
        for scale in (1e-11, 1, 1e12):
            for method in ("enumeration", "elimination"):
                with self.subTest(scale=scale, method=method):
                    # The original preferences, rescaled: take is clearly best after a rainy forecast.
                    data = load_example("umbrella")
                    for row in data["utility_nodes"][0]["table"]:
                        row["value"] *= scale
                    result = DecisionAnalysis(Network(data), method).analyze({"Forecast": "rainy"})
                    self.assertEqual(result["optimal_actions"], ["take"])
                    # 7 * 12/19 - 12 * 7/19 = 0: an exact tie, so both actions are optimal at any scale.
                    for row in data["utility_nodes"][0]["table"]:
                        row["value"] = tie[(row["when"]["Rain"], row["when"]["Umbrella"])] * scale
                    result = DecisionAnalysis(Network(data), method).analyze({"Forecast": "rainy"})
                    self.assertEqual(result["optimal_actions"], ["take", "leave"])


if __name__ == "__main__":
    unittest.main()
