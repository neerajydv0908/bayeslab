"""Run the documented umbrella demonstration, or --self-test for all checks."""

import argparse
from pathlib import Path
import unittest

from bayesnet import DecisionAnalysis, Enumeration, Network
from bayesnet.examples import load_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        root = Path(__file__).resolve().parent
        suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    network = Network(load_example("umbrella"))
    engine = Enumeration(network)
    print("UMBRELLA DECISION · fictional teaching model\n")
    print("Rain   Forecast   WetRoad   Joint probability")
    for row in engine.worlds():
        w = row["assignment"]
        print(f"{w['Rain']:<7}{w['Forecast']:<11}{w['WetRoad']:<10}{row['probability']:.3f}")
    print(f"\nP(Forecast=rainy) = {engine.event_probability({'Forecast': 'rainy'}):.2f}")
    print(f"P(WetRoad=yes) = {engine.event_probability({'WetRoad': 'yes'}):.2f}")
    for forecast in ("rainy", "dry"):
        evidence = {"Forecast": forecast}
        result = DecisionAnalysis(network).analyze(evidence)
        print(f"\nForecast: {forecast}")
        print(f"Probability of rain: {engine.probability({'Rain': 'yes'}, evidence):.2%}")
        for row in result["actions"]:
            print(f"EU({row['action']}): {row['expected_utility']:.6f}")
        print(f"Recommended action: {result['best_action']}; MEU = {result['maximum_expected_utility']:.6f}")
    before = DecisionAnalysis(network).analyze()
    forecast = before["information_values"][0]
    print(f"\nBefore the forecast: best action {before['best_action']}, MEU = {before['maximum_expected_utility']:.6f}")
    print(f"Value of observing the forecast (free): {forecast['value']:.2f} utility units")
    print("\nWith these preferences, take the umbrella above 20% rain probability; at 20%, both actions tie.")


if __name__ == "__main__":
    main()
