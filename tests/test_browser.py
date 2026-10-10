"""End-to-end checks of the web interface in a real (headless) browser.

Optional: needs Playwright and Chromium, which the core app does not.
    python -m pip install playwright && python -m playwright install chromium
    python -m unittest tests.test_browser -v
Skipped automatically when either is missing.
"""

from http.server import ThreadingHTTPServer
import importlib.util
import json
import os
import threading
import unittest

from bayesnet.examples import load_example
from bayesnet.server import Handler

HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None
# Continuous integration sets this so a missing browser fails instead of skipping.
REQUIRED = os.environ.get("BAYESLAB_REQUIRE_BROWSER") == "1"


class QuietHandler(Handler):
    def log_message(self, *args):
        pass


def three_action_model():
    model = load_example("umbrella")
    model["name"] = "Three ways to stay dry"
    model["decision_nodes"][0]["states"] = ["take", "leave", "stay"]
    model["utility_nodes"][0]["table"] += [{"when": {"Rain": r, "Umbrella": "stay"}, "value": 1} for r in ("yes", "no")]
    return model


@unittest.skipUnless(HAS_PLAYWRIGHT or REQUIRED, "Optional Playwright browser tests are not installed")
class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import Error, sync_playwright

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/"
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch()
        except Error as exc:  # Playwright is installed but its browser is not
            cls.playwright.stop()
            cls.server.shutdown()
            if REQUIRED:
                raise
            raise unittest.SkipTest(f"Chromium for Playwright is not available: {exc}")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1400, "height": 1000})
        self.page = self.context.new_page()
        self.page.set_default_timeout(10_000)
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.url)
        self.page.wait_for_selector(".big-result")

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [], "JavaScript errors on the page")

    def click(self, selector):
        self.page.click(selector)

    def text(self, selector):
        return self.page.inner_text(selector)

    def open_json_editor(self):
        self.click('[data-tab="model"]')
        self.page.wait_for_selector(".editor")
        self.click('[data-view="json"]')
        self.page.wait_for_selector("#model-editor")

    def apply_json(self, model):
        self.open_json_editor()
        self.page.fill("#model-editor", json.dumps(model))
        self.click('[data-command="apply"]')
        self.page.wait_for_selector(".editor-status:has-text('validated and applied')")

    def test_probability_queries_and_worlds(self):
        self.assertEqual(self.text(".big-result"), "63.16%")
        self.page.select_option("#method-select", "enumeration")
        self.click('[data-command="calculate"]')
        self.page.wait_for_selector(".success-label:has-text('enumeration')")
        self.assertEqual(self.text(".big-result"), "63.16%")
        self.click('[data-command="worlds"]')
        self.page.wait_for_selector(".worlds-table")
        self.assertEqual(self.page.locator(".worlds-table tbody tr").count(), 8)
        self.click('[data-mode="joint"]')
        self.click('[data-command="calculate"]')
        self.page.wait_for_selector(".big-result")
        self.assertEqual(self.text(".big-result"), "24.00%")

    def test_decision_value_of_information_and_costs(self):
        self.click('[data-tab="utility"]')
        self.page.wait_for_selector(".action-table")
        self.assertEqual(self.text(".utility-score .big-result"), "4.3158")
        self.click('[data-command="full-policy"]')
        self.page.wait_for_selector(".voi-table")
        self.assertIn("+2.24", self.text(".voi-table"))
        self.assertIn("Worth observing", self.text(".voi-table"))
        self.page.fill('[data-cost="Forecast"]', "3")
        self.click('[data-command="calculate"]')
        self.page.wait_for_selector(".voi-table")
        self.assertIn("Costs more than it is worth", self.text(".voi-table"))
        self.assertIn("-0.76", self.text(".policy-metrics"))
        # Changing a preference changes the recommendation and moves the crossover.
        self.page.fill('[aria-label="Comfort: Rain = no, Umbrella = leave"]', "40")
        self.click('[data-command="calculate"]')
        self.page.wait_for_selector(".sensitivity")
        self.assertIn("leave", self.text(".best-action"))
        self.assertIn("60.00%", self.text(".sensitivity"))

    def test_sensitivity_chart_with_three_actions(self):
        self.apply_json(three_action_model())
        self.click('[data-tab="utility"]')
        self.page.wait_for_selector(".sensitivity")
        message = self.text(".sensitivity p")
        self.assertIn("Best action by probability", message)
        self.assertIn("stay", message)
        self.assertEqual(self.page.locator(".sensitivity-legend span").count(), 3)
        self.assertEqual(self.page.locator('.sensitivity-svg path[stroke-dasharray="4 4"]').count(), 2)

    def test_visual_builder_creates_a_working_network(self):
        self.click('[data-tab="model"]')
        self.page.wait_for_selector(".builder")
        self.click('[data-b="add-chance"]')
        name = self.page.locator('[data-bfield="node-name"]')
        name.fill("Wind")
        name.press("Enter")
        self.page.wait_for_selector('.builder-node.active:has-text("Wind")')
        self.page.wait_for_selector(".draft-badge")
        # Make Wind depend on Rain, then give it an invalid row and normalize it.
        self.page.check('[data-bfield="parent"][data-parent="Rain"]')
        self.page.wait_for_selector('.builder-table th:has-text("Rain")')
        first = self.page.locator('[data-bprob][data-row="0"]').first
        first.fill("2")
        self.page.wait_for_selector('[data-sum-row="0"].bad')
        self.click('[data-b="normalize-row"][data-row="0"]')
        self.page.wait_for_selector('[data-sum-row="0"].ok')
        self.assertEqual(self.page.locator(".network-svg .node-group").count(), 6)
        self.click('[data-command="apply"]')
        self.page.wait_for_selector(".editor-status:has-text('validated and applied')")
        self.assertEqual(self.page.locator(".draft-badge").count(), 0)
        self.click('[data-tab="probability"]')
        self.click('[data-mode="marginal"]')
        self.page.select_option('[data-kind="event"][data-field="variable"]', "Wind")
        self.click('[data-command="calculate"]')
        self.page.wait_for_selector(".formula:has-text('Wind')")
        # Rain=yes row (2, 0.5) was normalized to (0.8, 0.2); Rain=no kept (0.5, 0.5): 0.3*0.8 + 0.7*0.5.
        self.assertEqual(self.text(".big-result"), "59.00%")

    def test_json_errors_and_discarding_changes(self):
        self.open_json_editor()
        self.page.fill("#model-editor", "{not json")
        self.click('[data-command="apply"]')
        self.page.wait_for_selector(".error-message")
        self.assertIn("not valid JSON", self.text(".error-message"))
        self.click('[data-view="builder"]')
        self.assertIn("builder cannot open it", self.text(".error-message"))
        self.click('[data-command="discard"]')
        self.page.wait_for_selector(".editor-status:has-text('discarded')")
        self.assertIn('"name": "The umbrella decision"', self.page.input_value("#model-editor"))

    def test_session_survives_a_reload(self):
        self.apply_json(three_action_model())
        self.page.reload()
        self.page.wait_for_selector(".card-title h2:has-text('Three ways to stay dry')")
        self.page.wait_for_selector("#toast.visible:has-text('Restored your last session')")
        self.click('[data-example="umbrella"]')
        self.page.wait_for_selector(".card-title h2:has-text('The umbrella decision')")

    def test_diagram_zoom_and_layout(self):
        svg = self.page.locator(".network-svg")
        before = float(svg.get_attribute("width"))
        self.click('[data-zoom="in"]')
        self.assertAlmostEqual(float(svg.get_attribute("width")), before * 1.25, delta=2)
        self.click('[data-zoom="out"]')
        self.click('[data-zoom="out"]')
        self.assertLess(float(svg.get_attribute("width")), before)
        # A long arrow (Rain -> Comfort) is routed through the layers, not under the whole graph.
        paths = self.page.locator(".network-svg path[marker-end]")
        self.assertEqual(paths.count(), 5)
        boxes = self.page.evaluate("""() => [...document.querySelectorAll('.network-svg .node-group')]
            .map((g) => { const m = g.transform.baseVal[0].matrix; return [m.e, m.f]; })""")
        self.assertEqual(len({tuple(box) for box in boxes}), 5)  # no two nodes share a position

    def import_model(self, model):
        self.click('[data-tab="model"]')
        self.page.wait_for_selector(".editor")
        self.page.set_input_files("#file-input", files=[{
            "name": "model.json", "mimeType": "application/json", "buffer": json.dumps(model).encode()}])

    def test_first_click_after_editing_a_field_is_not_lost(self):
        self.click('[data-tab="model"]')
        self.page.wait_for_selector(".builder")
        self.page.fill('[data-bfield="node-name"]', "Rainfall")  # still focused: no change event yet
        # A person holds the button for a moment; the field loses focus (and changes) on mousedown.
        self.page.click('[data-command="apply"]', delay=120)  # one click must blur the field *and* apply
        self.page.wait_for_selector(".editor-status:has-text('validated and applied')")
        self.assertEqual(self.page.locator('.builder-node.active strong').inner_text(), "Rainfall")
        self.page.fill('[data-bprob][data-row="0"][data-state="yes"]', "0.6")
        self.page.click('[data-b="normalize-row"][data-row="0"]', delay=120)
        self.page.wait_for_selector('[data-sum-row="0"].ok')
        # (0.6, 0.7) scaled to add up to 1
        self.assertEqual(self.page.input_value('[data-bprob][data-row="0"][data-state="no"]'), "0.538461538462")

    def test_imported_text_is_never_rendered_as_html(self):
        model = load_example("umbrella")
        model["utility_nodes"][0]["table"][0]["value"] = '<b id="pwned">x</b>'
        model["utility_nodes"][0]["table"][1]["value"] = None
        self.import_model(model)
        self.page.wait_for_selector(".builder")
        self.page.locator(".network-svg .node-group", has_text="Comfort").click()
        self.page.wait_for_selector("#inspector h3:has-text('Comfort')")
        self.assertEqual(self.page.locator("#pwned").count(), 0)
        self.assertIn('<b id="pwned">', self.text("#inspector"))

    def test_missing_rows_are_filled_visibly_not_silently(self):
        model = load_example("umbrella")
        model["chance_nodes"][1]["cpt"].pop()  # Forecast loses its Rain = no row
        model["utility_nodes"][0]["table"].pop()
        self.import_model(model)
        self.page.wait_for_selector(".editor-status:has-text('filled with defaults')")
        status = self.text(".editor-status")
        self.assertIn("Forecast (1 row)", status)
        self.assertIn("Comfort (1 row)", status)
        self.assertEqual(self.page.locator(".builder-node .issue").count(), 2)
        self.click('[data-b="select"][data-name="Forecast"]')
        self.page.wait_for_selector(".builder-table tr.filled")
        self.assertIn("filled with defaults", self.text(".builder-warning"))
        self.click('[data-b="normalize-row"][data-row="1"]')  # reviewing the row clears the flag
        self.page.wait_for_selector(".builder-node[data-name='Forecast']:not(:has(.issue))")
        self.assertEqual(self.page.locator(".builder-table tr.filled").count(), 0)

    def test_session_is_kept_when_the_server_is_busy_during_a_reload(self):
        self.apply_json(three_action_model())
        self.click('[data-tab="probability"]')  # restoring this tab recalculates straight away
        self.page.route("**/api/query", lambda route: route.fulfill(
            status=503, content_type="application/json", body='{"error": "busy"}'))
        self.page.reload()
        self.page.wait_for_selector(".card-title h2:has-text('Three ways to stay dry')")
        self.page.wait_for_selector(".error-message:has-text('busy')")
        self.page.unroute("**/api/query")
        self.page.reload()
        self.page.wait_for_selector(".card-title h2:has-text('Three ways to stay dry')")
        self.page.wait_for_selector(".big-result")

    def test_deleting_every_node_leaves_a_usable_page(self):
        self.click('[data-tab="model"]')
        self.page.wait_for_selector(".builder")
        while self.page.locator('[data-b="delete"]').count():
            self.click('[data-b="delete"]')
        self.page.wait_for_selector(".graph-empty:has-text('No nodes yet')")
        self.click('[data-b="add-chance"]')
        self.page.wait_for_selector(".network-svg .node-group")
        self.click('[data-command="discard"]')
        self.page.wait_for_selector(".card-title h2:has-text('The umbrella decision')")

    def test_phone_layout_has_no_sideways_scroll(self):
        page = self.context.new_page()
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(self.url)
        page.wait_for_selector(".big-result")
        for tab in ("probability", "utility", "model"):
            page.click(f'[data-tab="{tab}"]')
            page.wait_for_timeout(300)
            width, viewport = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
            self.assertLessEqual(width, viewport, tab)
        page.close()


if __name__ == "__main__":
    unittest.main()
