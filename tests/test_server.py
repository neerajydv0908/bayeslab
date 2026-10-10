"""Exercise real HTTP request parsing and routes without binding a socket."""

from io import BytesIO
import json
import unittest
from unittest.mock import patch

from bayesnet.examples import load_example
from bayesnet.server import Handler


class MemorySocket:
    def __init__(self, data):
        self.input = BytesIO(data)
        self.output = bytearray()

    def makefile(self, *args, **kwargs):
        return self.input

    def sendall(self, data):
        self.output.extend(data)


class QuietHandler(Handler):
    def log_message(self, *args):
        pass


def request(path, body=None, raw=None):
    method = "POST" if body is not None or raw is not None else "GET"
    payload = raw if raw is not None else json.dumps(body).encode() if body is not None else b""
    headers = f"{method} {path} HTTP/1.0\r\nHost: localhost\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
    socket = MemorySocket(headers + payload)
    QuietHandler(socket, ("127.0.0.1", 1234), object())
    head, content = bytes(socket.output).split(b"\r\n\r\n", 1)
    status = int(head.split(b" ")[1])
    return status, head, content


class ServerTests(unittest.TestCase):
    def test_static_app_and_example_routes(self):
        for path in ("/", "/app.js", "/styles.css", "/api/examples", "/api/examples/umbrella"):
            with self.subTest(path=path):
                status, headers, body = request(path)
                self.assertEqual(status, 200)
                self.assertTrue(body)
                self.assertIn(b"nosniff", headers)
        self.assertEqual(request("/../model.py")[0], 404)
        self.assertEqual(request("/api/examples/unknown")[0], 404)

    def test_query_endpoint(self):
        status, _, body = request("/api/query", {"model":load_example("alarm"), "event":{"Burglary":"yes"}, "evidence":{"JohnCalls":"yes", "MaryCalls":"yes"}})
        self.assertEqual(status, 200)
        self.assertAlmostEqual(json.loads(body)["probability"], .2841718353643929)

    def test_utility_and_validation_endpoints(self):
        body = {"model":load_example("umbrella")}
        status, _, result = request("/api/utility", body)
        self.assertEqual(status, 200)
        self.assertAlmostEqual(json.loads(result)["policy_expected_utility"], 3.24)
        status, _, result = request("/api/validate", body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(result)["decision_nodes"], 1)

    def test_invalid_json_and_models_return_useful_errors(self):
        for raw in (b"{", b"[]", b'{}', b'{"model": null}'):
            with self.subTest(raw=raw):
                status, _, result = request("/api/query", raw=raw)
                self.assertEqual(status, 400)
                self.assertTrue(json.loads(result)["error"])

    def test_deeply_nested_json_gets_a_400_reply(self):
        raw = b'{"model": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"
        status, _, result = request("/api/query", raw=raw)
        self.assertEqual(status, 400)
        self.assertIn("nested too deeply", json.loads(result)["error"])

    def test_utilities_too_large_get_a_400(self):
        model = load_example("umbrella")
        for row in model["utility_nodes"][0]["table"]:
            row["value"] = 1e308
        model["utility_nodes"].append({"name": "Extra", "parents": [], "table": [{"when": {}, "value": 1e308}]})
        for path in ("/api/validate", "/api/utility", "/api/sensitivity"):
            with self.subTest(path=path):
                status, _, result = request(path, {"model": model, "target": "Rain"})
                self.assertEqual(status, 400)
                self.assertIn("too large", json.loads(result)["error"])

    def test_unexpected_errors_still_get_a_json_reply(self):
        with patch("bayesnet.server.Network", side_effect=RuntimeError("internal detail")):
            status, _, result = request("/api/validate", {"model": load_example("umbrella")})
        self.assertEqual(status, 500)
        self.assertNotIn("internal detail", json.loads(result)["error"])
        with patch("bayesnet.server.load_example", side_effect=RuntimeError("internal detail")):
            status, _, result = request("/api/examples/umbrella")
        self.assertEqual(status, 500)
        self.assertTrue(json.loads(result)["error"])


if __name__ == "__main__":
    unittest.main()

