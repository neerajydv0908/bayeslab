"""Tests for version 1.2: limits, value of information, sensitivity, hosting and errors."""

from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bayesnet import ComputationTimeout, DecisionAnalysis, Enumeration, Inference, ModelError, Network, time_limit
from bayesnet.__main__ import read_model, server_options
from bayesnet.decision import TINY_BRANCH
from bayesnet.examples import load_example
from bayesnet.inference import FactorTooLarge
from bayesnet.server import ServerSettings
from tests.test_decision import observed_chain
from tests.test_server import MemorySocket, QuietHandler, request


def binary_chain(size, observed_prefix=0):
    nodes = []
    for i in range(size):
        parents = [f"X{i-1}"] if i else []
        rows = [{"when": dict(zip(parents, combo)), "probabilities": {"yes": .7 if "yes" in combo else .2,
                                                                       "no": .3 if "yes" in combo else .8}}
                for combo in product(*(("yes", "no") for _ in parents))]
        if not parents:
            rows = [{"when": {}, "probabilities": {"yes": .4, "no": .6}}]
        nodes.append({"name": f"X{i}", "states": ["yes", "no"], "parents": parents, "cpt": rows})
    return Network({"name": "chain", "chance_nodes": nodes})


def xor_model():
    """C = A xor B: neither A nor B alone says anything about C, but both reveal it."""
    coin = {"yes": .5, "no": .5}
    xor = [{"when": {"A": a, "B": b}, "probabilities": {"yes": float(a != b), "no": float(a == b)}}
           for a, b in product(("yes", "no"), repeat=2)]
    return {"chance_nodes": [
        {"name": "A", "states": ["yes", "no"], "parents": [], "cpt": [{"when": {}, "probabilities": coin}]},
        {"name": "B", "states": ["yes", "no"], "parents": [], "cpt": [{"when": {}, "probabilities": coin}]},
        {"name": "C", "states": ["yes", "no"], "parents": ["A", "B"], "cpt": xor}],
        "decision_nodes": [{"name": "Guess", "states": ["yes", "no"], "information": ["A", "B"]}],
        "utility_nodes": [{"name": "Right", "parents": ["C", "Guess"], "table": [
            {"when": {"C": c, "Guess": g}, "value": float(c == g)} for c, g in product(("yes", "no"), repeat=2)]}]}


class EnumerationLimitTests(unittest.TestCase):
    def test_limit_counts_only_worlds_that_agree_with_the_evidence(self):
        network = binary_chain(24)  # 16,777,216 worlds in total
        evidence = {f"X{i}": "yes" for i in range(1, 23)}
        result = Enumeration(network).query({"X0": "yes"}, evidence)
        self.assertAlmostEqual(result["probability"], Inference(network).probability({"X0": "yes"}, evidence), places=12)
        self.assertIn("of 16,777,216 possible", result["trace"][0]["description"])
        with self.assertRaisesRegex(ModelError, "Enumeration exceeds.*agree with this evidence"):
            Enumeration(network).query({"X0": "yes"}, {"X1": "yes"})
        with self.assertRaisesRegex(ModelError, "Enumeration exceeds"):
            list(Enumeration(network).worlds())  # the full joint table still has every world


class TimeLimitTests(unittest.TestCase):
    def test_long_calculations_stop_at_the_limit(self):
        network = binary_chain(20)
        with self.assertRaisesRegex(ComputationTimeout, "0.001-second time limit"):
            with time_limit(0.001):
                Enumeration(network).probability({"X19": "yes"})
        # Without a limit (and after the block) the same calculation completes.
        self.assertGreater(Enumeration(network).probability({"X19": "yes"}), 0)

    def test_nested_limits_only_shorten_and_invalid_limits_are_rejected(self):
        with time_limit(0.001):
            with time_limit(60):
                with self.assertRaises(ComputationTimeout):
                    Enumeration(binary_chain(20)).probability({"X19": "yes"})
        for bad in (0, -1, float("inf"), "5", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                with time_limit(bad):
                    pass
        with time_limit(None):
            self.assertAlmostEqual(Inference(binary_chain(3)).probability({"X0": "yes"}), .4)


class InformationValueTests(unittest.TestCase):
    def test_umbrella_forecast_value_and_cost(self):
        data = load_example("umbrella")
        for method in ("elimination", "enumeration"):
            result = DecisionAnalysis(Network(data), method).analyze()
            (forecast,) = result["information_values"]
            self.assertEqual(forecast["variable"], "Forecast")
            self.assertAlmostEqual(forecast["value"], 2.24)
            self.assertEqual(forecast["cost"], 0)
            self.assertTrue(forecast["worth_observing"])
        data["decision_nodes"][0]["observation_costs"] = {"Forecast": 3}
        result = DecisionAnalysis(Network(data)).analyze()
        self.assertAlmostEqual(result["information_values"][0]["net_value"], -0.76)
        self.assertFalse(result["information_values"][0]["worth_observing"])
        self.assertAlmostEqual(result["net_value_of_information"], -0.76)
        self.assertEqual(result["observation_cost"], 3)
        # Once the forecast is observed there is nothing left to value or pay for.
        observed = DecisionAnalysis(Network(data)).analyze({"Forecast": "dry"})
        self.assertEqual((observed["information_values"], observed["observation_cost"]), ([], 0))

    def test_variables_can_be_worthless_alone_but_valuable_together(self):
        for method in ("elimination", "enumeration"):
            result = DecisionAnalysis(Network(xor_model()), method).analyze()
            self.assertAlmostEqual(result["maximum_expected_utility"], .5)
            self.assertEqual([row["value"] for row in result["information_values"]], [0.0, 0.0])
            self.assertAlmostEqual(result["value_of_information"], .5)

    def test_observation_costs_are_validated(self):
        for costs, message in (({"Rain": 1}, "not in its information set"), ({"Forecast": -1}, "zero or positive"),
                               ({"Forecast": "free"}, "finite number"), ([1], "must be an object")):
            data = load_example("umbrella")
            data["decision_nodes"][0]["observation_costs"] = costs
            with self.subTest(costs=costs), self.assertRaisesRegex(ModelError, message):
                Network(data)


class SharedPolicyTests(unittest.TestCase):
    def test_shared_factors_match_one_calculation_per_branch(self):
        for action_dependent in (False, True):
            network = Network(observed_chain(size=9, observed=5, action_dependent=action_dependent))
            fast = DecisionAnalysis(network).analyze()
            with patch.object(DecisionAnalysis, "_shared_branch_rows", side_effect=FactorTooLarge("forced")):
                slow = DecisionAnalysis(network).analyze()
            with self.subTest(action_dependent=action_dependent):
                for row, other in zip(fast["policy"], slow["policy"]):
                    self.assertEqual(row["optimal_actions"], other["optimal_actions"])
                    for a, b in zip(row["actions"], other["actions"]):
                        self.assertAlmostEqual(a["expected_utility"], b["expected_utility"], places=12)
                for key in ("policy_expected_utility", "value_of_information"):
                    self.assertAlmostEqual(fast[key], slow[key], places=12)

    def test_extremely_unlikely_branches_are_recalculated_exactly(self):
        data = load_example("umbrella")
        # A forecast that says "dry" with probability around 1e-300 when it rains and never otherwise.
        for row in data["chance_nodes"][1]["cpt"]:
            row["probabilities"] = ({"rainy": 1 - 1e-300, "dry": 1e-300} if row["when"]["Rain"] == "yes"
                                    else {"rainy": 1.0, "dry": 0.0})
        network = Network(data)
        elimination = DecisionAnalysis(network).analyze()
        enumeration = DecisionAnalysis(network, "enumeration").analyze()
        dry = [row for row in elimination["policy"] if row["when"]["Forecast"] == "dry"][0]
        self.assertLess(dry["probability"], TINY_BRANCH)
        self.assertEqual(dry["action"], "take")  # a dry forecast here only happens when it rains
        self.assertAlmostEqual(dry["expected_utility"], 8)
        for row, other in zip(elimination["policy"], enumeration["policy"]):
            self.assertEqual(row["optimal_actions"], other["optimal_actions"])
            self.assertAlmostEqual(row["expected_utility"], other["expected_utility"], places=9)


class SensitivityTests(unittest.TestCase):
    def three_actions(self):
        data = load_example("umbrella")
        data["decision_nodes"][0]["states"] = ["take", "leave", "stay"]
        data["utility_nodes"][0]["table"] += [{"when": {"Rain": r, "Umbrella": "stay"}, "value": 1} for r in ("yes", "no")]
        return Network(data)

    def test_upper_envelope_for_three_actions(self):
        # take: 10p - 2, leave: 5 - 25p, stay: 1. Leave is best below 0.16, stay up to 0.3, take above.
        result = DecisionAnalysis(self.three_actions()).sensitivity("Rain")
        self.assertEqual([r["optimal_actions"] for r in result["regions"]], [["leave"], ["stay"], ["take"]])
        self.assertEqual(len(result["thresholds"]), 2)
        for got, expected in zip(result["thresholds"], (.16, .3)):
            self.assertAlmostEqual(got, expected)
        self.assertIsNone(result["threshold"])
        self.assertEqual(set(result["points"][0]["actions"]), {"take", "leave", "stay"})
        for region in result["regions"]:
            middle = Fraction(region["from"] + region["to"]) / 2
            values = {line["action"]: line["intercept"] + line["slope"] * float(middle) for line in result["lines"]}
            self.assertEqual(region["optimal_actions"], [max(values, key=values.get)])

    def test_two_actions_keep_the_single_threshold(self):
        result = DecisionAnalysis(Network(load_example("umbrella"))).sensitivity("Rain")
        self.assertAlmostEqual(result["threshold"], .2)
        self.assertEqual(len(result["thresholds"]), 1)
        self.assertAlmostEqual(result["thresholds"][0], .2)
        self.assertEqual([r["optimal_actions"] for r in result["regions"]], [["leave"], ["take"]])


class ServerAndCliTests(unittest.TestCase):
    def test_health_config_and_builder_routes(self):
        status, _, body = request("/healthz")
        self.assertEqual((status, json.loads(body)), (200, {"status": "ok"}))
        status, _, body = request("/api/config")
        config = json.loads(body)
        self.assertEqual(status, 200)
        self.assertFalse(config["public"])
        self.assertEqual(config["time_limit_seconds"], 60)
        status, headers, body = request("/builder.js")
        self.assertEqual(status, 200)
        self.assertIn(b"text/javascript", headers)
        self.assertIn(b"const Builder", body)

    def test_request_errors_are_explained(self):
        cases = ((b"{", "not valid JSON: Expecting property name"), (b'{"event": {}}', 'Include the network as "model"'),
                 (b"\xff\xfe", "UTF-8"))
        for raw, message in cases:
            with self.subTest(raw=raw):
                status, _, body = request("/api/query", raw=raw)
                self.assertEqual(status, 400)
                self.assertIn(message, json.loads(body)["error"])
        payload = b'{"model": {}}'
        head = "POST /api/query HTTP/1.0\r\nHost: x\r\nContent-Length: abc\r\n\r\n".encode()
        socket = MemorySocket(head + payload)
        QuietHandler(socket, ("127.0.0.1", 1), object())
        self.assertIn(b"valid Content-Length header", bytes(socket.output))

    def test_public_settings_busy_server_and_time_limit(self):
        class TimedSocket(MemorySocket):
            def settimeout(self, value):
                self.timeout = value

        model = Network(observed_chain()).source
        busy = threading.BoundedSemaphore(1)
        busy.acquire()
        server = SimpleNamespace(bayeslab_settings=ServerSettings(public=True, time_limit=10, max_concurrent=1,
                                                                   queue_wait=0.01),
                                 bayeslab_slots=busy)
        raw = json.dumps({"model": model}).encode()
        socket = TimedSocket(f"POST /api/utility HTTP/1.0\r\nContent-Length: {len(raw)}\r\n\r\n".encode() + raw)
        QuietHandler(socket, ("127.0.0.1", 1), server)
        self.assertEqual(socket.timeout, 30)  # public mode drops stalled clients
        self.assertIn(b"503", bytes(socket.output).split(b"\r\n")[0])
        self.assertIn(b"Retry-After: 5", bytes(socket.output))
        server.bayeslab_slots = threading.BoundedSemaphore(1)
        server.bayeslab_settings = ServerSettings(public=True, time_limit=0.0005)
        heavy = json.dumps({"model": binary_chain(20).source, "event": {"X19": "yes"},
                            "method": "enumeration"}).encode()
        socket = TimedSocket(f"POST /api/query HTTP/1.0\r\nContent-Length: {len(heavy)}\r\n\r\n".encode() + heavy)
        QuietHandler(socket, ("127.0.0.1", 1), server)
        self.assertIn(b"400", bytes(socket.output).split(b"\r\n")[0])
        self.assertIn(b"time limit", bytes(socket.output))

    def test_server_options_follow_hosting_environment(self):
        args = SimpleNamespace(host=None, port=None, public=False, time_limit=None)
        self.assertEqual(server_options(args, {}), {"host": "127.0.0.1", "port": 8000, "public": False, "time_limit": None})
        hosted = server_options(args, {"PORT": "10000", "BAYESLAB_PUBLIC": "1", "BAYESLAB_TIME_LIMIT": "5"})
        self.assertEqual(hosted, {"host": "0.0.0.0", "port": 10000, "public": True, "time_limit": 5.0})
        explicit = SimpleNamespace(host="127.0.0.1", port=9000, public=True, time_limit=2.0)
        self.assertEqual(server_options(explicit, {"PORT": "10000"})["port"], 9000)
        self.assertEqual(server_options(explicit, {})["host"], "127.0.0.1")
        for env, message in (({"PORT": "web"}, "PORT must be a whole number"),
                             ({"BAYESLAB_TIME_LIMIT": "soon"}, "number of seconds"),
                             ({"BAYESLAB_TIME_LIMIT": "inf"}, "finite number of seconds"),
                             ({"BAYESLAB_TIME_LIMIT": "nan"}, "finite number of seconds")):
            with self.subTest(env=env), self.assertRaisesRegex(ModelError, message):
                server_options(args, env)

    def test_model_files_are_explained(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder, "bad.json")
            bad.write_text('{"name": ', encoding="utf-8")
            with self.assertRaisesRegex(ModelError, r"bad.json is not valid JSON: Expecting value \(line 1"):
                read_model(bad)
            with self.assertRaisesRegex(ModelError, "Model file not found"):
                read_model(Path(folder, "missing.json"))
            with self.assertRaisesRegex(ModelError, "is a folder"):
                read_model(Path(folder))
            good = Path(folder, "good.json")
            good.write_text(json.dumps(load_example("alarm")), encoding="utf-8")
            self.assertEqual(Network(read_model(good)).summary()["chance_nodes"], 5)


if __name__ == "__main__":
    unittest.main()
