"""Expected utility, maximum expected utility, optimal policies, and the value of information."""

from itertools import product
import math

from .enumeration import Enumeration
from .inference import FactorTooLarge, ImpossibleEvidence, Inference
from .limits import check_time
from .model import ModelError, values_of
from .numeric import ScaledSum

MAX_POLICY_ROWS = 512
# Two expected utilities tie when they differ by at most this fraction of the
# model's utility scale: the largest absolute total utility any outcome can
# produce. Unlike a fixed absolute tolerance, this gives the same ties after
# every utility is multiplied by the same positive constant.
TIE_TOLERANCE = 1e-12
# Elimination policies share one factor per action and utility across all
# observation branches. A branch less likely than this (relative to the
# evidence) is recalculated on its own, so its expected utilities never rest on
# products near the bottom of the float range.
TINY_BRANCH = 1e-250


class _Recalculate(Exception):
    """The shared elimination factors cannot serve every branch; use one calculation per branch."""


class DecisionAnalysis:
    def __init__(self, network, method="elimination"):
        if not network.decision:
            raise ModelError("Utility analysis requires a decision node and a utility table.")
        self.network = network
        self.engine = Inference(network)
        if method not in ("enumeration", "elimination"):
            raise ModelError("Choose enumeration or elimination.")
        self.method = method
        # Largest absolute total utility any outcome can produce: the tie scale.
        self.utility_scale = network.utility_scale

    def _evidence(self, evidence):
        evidence = {} if evidence is None else evidence
        decision = self.network.decision
        if isinstance(evidence, dict):
            for name in evidence:
                if name in self.network.chance and name not in decision.information:
                    observed = ", ".join(decision.information) or "nothing"
                    raise ModelError(f"{name} is not observed before the {decision.name} decision, so it cannot be "
                                     f"decision evidence. Observed before deciding: {observed}.")
        self.network.validate_observations(evidence, "Decision evidence", decision.information)
        return evidence

    def expected_utility(self, action, evidence=None):
        evidence = self._evidence(evidence)
        self.network.validate_action(action)
        if self.method == "enumeration":
            return self._rows(self._enumerate(evidence, (), (action,))[()])[0]
        components = []
        for utility in self.network.utilities:
            distribution, _ = self.engine._distribution(utility.parents, evidence, action)
            value = math.fsum(probability * utility.table[key] for key, probability in distribution.values.items())
            components.append({"name": utility.name, "expected_utility": value})
        return {"action": action, "expected_utility": math.fsum(row["expected_utility"] for row in components), "components": components}

    def sensitivity(self, target, evidence=None):
        """Vary a binary outcome's posterior without changing the saved CPTs.

        Works for any number of actions, provided the actions do not change
        chance probabilities and every utility depends only on this outcome and
        the decision. Each action's expected utility is then a straight line in
        the outcome's probability p; the best action follows the upper envelope
        of those lines, which changes at the reported thresholds.
        """
        evidence = self._evidence(evidence)
        network, decision = self.network, self.network.decision
        if target not in network.chance_names or len(network.domains[target]) != 2:
            raise ModelError("Sensitivity requires a binary chance variable.")
        if any(decision.name in n.parents for n in network.chance.values()):
            raise ModelError("Sensitivity requires actions that do not change chance probabilities.")
        if any(set(u.parents) - {target, decision.name} for u in network.utilities):
            raise ModelError("Sensitivity utilities must depend only on the selected outcome and decision.")
        positive, negative = network.domains[target]
        lines = []
        for action in decision.states:
            scores = {}
            for state in (positive, negative):
                assignment = {target: state, decision.name: action}
                scores[state] = math.fsum(u.table[tuple(assignment[p] for p in u.parents)] for u in network.utilities)
            lines.append({"action": action, "intercept": scores[negative],
                          "slope": scores[positive] - scores[negative], "utilities": scores})
        regions = self._envelope(lines, TIE_TOLERANCE * self.utility_scale)
        thresholds = [region["from"] for region in regions[1:]]
        if len(lines) == 2:
            # The crossing of the two lines, kept for compatibility even at p = 0 or 1.
            slope_delta = lines[0]["slope"] - lines[1]["slope"]
            intercept_delta = lines[0]["intercept"] - lines[1]["intercept"]
            crossing = -intercept_delta / slope_delta if slope_delta != 0 else None
            threshold = crossing if crossing is not None and 0 <= crossing <= 1 else None
        else:
            threshold = thresholds[0] if len(thresholds) == 1 else None
        always_tied = all(line["slope"] == lines[0]["slope"] and line["intercept"] == lines[0]["intercept"]
                          for line in lines)
        posterior = self.engine.distribution(target, evidence, decision.states[0])[positive]
        return {"target": target, "state": positive, "other_state": negative,
                "posterior": posterior, "lines": lines, "threshold": threshold, "thresholds": thresholds,
                "regions": regions, "always_tied": always_tied,
                "points": [{"probability": i / 100, "actions": {
                    line["action"]: line["intercept"] + line["slope"] * i / 100 for line in lines
                }} for i in range(101)]}

    @staticmethod
    def _envelope(lines, tolerance):
        """Split [0, 1] into intervals with the same set of optimal actions.

        The optimal set can only change where two lines cross, so evaluating
        one point inside each interval between crossings is exact.
        """
        crossings = []
        for i, first in enumerate(lines):
            for second in lines[i + 1:]:
                slope_delta = first["slope"] - second["slope"]
                if slope_delta != 0:
                    crossing = (second["intercept"] - first["intercept"]) / slope_delta
                    if 0 < crossing < 1:
                        crossings.append(crossing)
        points = [0.0]
        for crossing in sorted(crossings):
            if crossing - points[-1] > 1e-12:
                points.append(crossing)
        if 1.0 - points[-1] > 1e-12:
            points.append(1.0)
        else:
            points[-1] = 1.0
        regions = []
        for low, high in zip(points, points[1:]):
            middle = (low + high) / 2
            values = [line["intercept"] + line["slope"] * middle for line in lines]
            best = max(values)
            optimal = [line["action"] for line, value in zip(lines, values) if best - value <= tolerance]
            if regions and regions[-1]["optimal_actions"] == optimal:
                regions[-1]["to"] = high
            else:
                regions.append({"from": low, "to": high, "optimal_actions": optimal})
        return regions

    def _enumerate(self, evidence, unknown, actions):
        """Visit each world that agrees with the evidence exactly once.

        Returns {branch: {action: (mass, weighted)}} for every assignment
        `branch` of the `unknown` variables: mass is P(branch, evidence |
        do(action)) and weighted[i] sums P(world | do(action)) * U_i(world,
        action) over the branch's worlds, all as ScaledSums. A whole policy
        therefore costs one depth-first walk over the worlds (one per action
        if the action changes a chance CPT), not one per branch and action.
        """
        network, decision_name = self.network, self.network.decision.name
        engine = Enumeration(network)
        engine._check_size(evidence)
        branch_of = values_of(unknown)
        utilities = [(values_of(utility.parents), utility.table) for utility in network.utilities]
        sums = {branch: {action: (ScaledSum(), [ScaledSum() for _ in utilities]) for action in actions}
                for branch in product(*(network.domains[name] for name in unknown))}

        def add(world, value, exponent, action):
            mass, weighted = sums[branch_of(world)][action]
            mass.add(value, exponent)
            for (utility_key, table), total in zip(utilities, weighted):
                total.add(value * table[utility_key(world)], exponent)

        if any(decision_name in node.parents for node in network.chance.values()):
            # The action changes chance probabilities, so each action needs its own walk.
            for action in actions:
                engine._walk(evidence, action,
                             lambda world, value, exponent, action=action: add(world, value, exponent, action))
        else:
            def leaf(world, value, exponent):
                for action in actions:
                    world[decision_name] = action  # read only by the utility tables
                    add(world, value, exponent, action)
                world[decision_name] = actions[0]

            engine._walk(evidence, actions[0], leaf)
        return sums

    @staticmethod
    def _combine(branches):
        """Add the per-branch sums together, giving the sums for the evidence alone."""
        cells = list(branches.values())
        return {action: (ScaledSum.combined(cell[action][0] for cell in cells),
                         [ScaledSum.combined(column) for column in zip(*(cell[action][1] for cell in cells))])
                for action in cells[0]}

    def _rows(self, sums):
        """Turn {action: (mass, weighted)} sums into expected-utility rows."""
        rows = []
        for action, (mass, weighted) in sums.items():
            if mass.is_zero():
                raise ImpossibleEvidence("The evidence has zero probability; expected utility is undefined.")
            components = [{"name": utility.name, "expected_utility": total.ratio(mass)}
                          for utility, total in zip(self.network.utilities, weighted)]
            rows.append({"action": action, "expected_utility": math.fsum(c["expected_utility"] for c in components),
                         "components": components})
        return rows

    def _row(self, action, components):
        return {"action": action, "expected_utility": math.fsum(c["expected_utility"] for c in components),
                "components": components}

    def _elimination_branches(self, evidence, unknown, actions):
        """Branch probabilities P(i | e) and expected-utility rows for every observation branch i.

        Instead of one elimination per branch, action and utility, this runs one
        elimination per action and utility over the unobserved information and
        that utility's parents, then reads every branch off the shared factor.
        Very unlikely branches, and factors too large to share, fall back to a
        separate calculation per branch.
        """
        states = self.network.decision.states
        observation, _ = self.engine._distribution(unknown, evidence, states[0])
        probabilities = dict(observation.values)
        if not unknown:
            return probabilities, {(): actions}  # the only branch is the evidence itself
        reachable = [branch for branch, probability in probabilities.items() if probability != 0]
        shared = [branch for branch in reachable if probabilities[branch] >= TINY_BRANCH]
        rows = {}
        try:
            rows = self._shared_branch_rows(evidence, unknown, shared)
        except (FactorTooLarge, _Recalculate):
            rows = {}
        for branch in reachable:
            if branch not in rows:
                check_time()
                observed = {**evidence, **dict(zip(unknown, branch))}
                rows[branch] = [self.expected_utility(action, observed) for action in states]
        return probabilities, rows

    def _shared_branch_rows(self, evidence, unknown, branches):
        if not branches:
            return {}
        network, size = self.network, len(unknown)
        states = network.decision.states
        components = {branch: [[] for _ in states] for branch in branches}
        for action_index, action in enumerate(states):
            for utility in network.utilities:
                check_time()
                targets = unknown + tuple(name for name in utility.parents if name not in unknown)
                positions = [targets.index(name) for name in utility.parents]
                distribution, _ = self.engine._distribution(targets, evidence, action)
                mass, weighted = {}, {}
                for key, probability in distribution.values.items():
                    if probability:
                        branch = key[:size]
                        mass.setdefault(branch, []).append(probability)
                        weighted.setdefault(branch, []).append(
                            probability * utility.table[tuple(key[i] for i in positions)])
                for branch in branches:
                    total = math.fsum(mass.get(branch, ()))
                    if total <= 0:
                        raise _Recalculate
                    components[branch][action_index].append(
                        {"name": utility.name, "expected_utility": math.fsum(weighted[branch]) / total})
        return {branch: [self._row(action, cells[i]) for i, action in enumerate(states)]
                for branch, cells in components.items()}

    def _optimal(self, rows):
        """The best expected utility and every action that ties with it, in declared order."""
        best = max(row["expected_utility"] for row in rows)
        tolerance = TIE_TOLERANCE * self.utility_scale
        return best, [row["action"] for row in rows if best - row["expected_utility"] <= tolerance]

    def _information_values(self, unknown, probabilities, rows, meu):
        """The value of observing each unobserved information variable on its own.

        For a variable X: VOI(X) = Σ_x P(x | e) max_a EU(a | e, x) − MEU(e).
        Branches assign every unobserved variable, so EU(a | e, x) is the
        probability-weighted average of the branch expected utilities with X = x.
        """
        tolerance = TIE_TOLERANCE * self.utility_scale
        costs = self.network.decision.observation_costs
        count = len(self.network.decision.states)
        results = []
        for index, name in enumerate(unknown):
            groups = {}
            for branch, probability in probabilities.items():
                if probability != 0:
                    groups.setdefault(branch[index], []).append((probability, rows[branch]))
            # max_a Σ_b P(b | e) EU(a | e, b) = P(x | e) max_a EU(a | e, x)
            best = [max(math.fsum(p * branch_rows[k]["expected_utility"] for p, branch_rows in members)
                        for k in range(count)) for members in groups.values()]
            gain = math.fsum(best) - meu
            value = gain if gain > tolerance else 0.0
            cost = costs.get(name, 0.0)
            results.append({"variable": name, "value": value, "cost": cost, "net_value": value - cost,
                            "worth_observing": value - cost > tolerance})
        return results

    def analyze(self, evidence=None):
        evidence = self._evidence(evidence)
        decision = self.network.decision
        unknown = tuple(name for name in decision.information if name not in evidence)
        if math.prod(len(self.network.domains[name]) for name in unknown) > MAX_POLICY_ROWS:
            raise ModelError(f"The observation policy exceeds {MAX_POLICY_ROWS} rows. Supply more evidence or simplify the information set.")
        # Information arcs precede the decision in the validated DAG, so the
        # distribution of the observations cannot depend on the selected action.
        if self.method == "enumeration":
            # One pass yields every branch; adding the branches gives the evidence alone.
            branches = self._enumerate(evidence, unknown, decision.states)
            actions = self._rows(self._combine(branches))
            first = decision.states[0]
            total = ScaledSum.combined(cells[first][0] for cells in branches.values())
            probabilities = {branch: cells[first][0].ratio(total) for branch, cells in branches.items()}
            branch_rows = {branch: self._rows(branches[branch])
                           for branch, probability in probabilities.items() if probability != 0}
        else:
            actions = [self.expected_utility(action, evidence) for action in decision.states]
            probabilities, branch_rows = self._elimination_branches(evidence, unknown, actions)
        meu, optimal = self._optimal(actions)
        policy = []
        for states in product(*(self.network.domains[name] for name in unknown)):
            probability = probabilities[states]
            observed = {**evidence, **dict(zip(unknown, states))}
            if probability == 0:
                policy.append({"when": observed, "probability": 0.0, "action": None,
                               "optimal_actions": [], "expected_utility": None})
                continue
            branch_actions = branch_rows[states]
            value, choices = self._optimal(branch_actions)
            # All ties are reported; the first declared action gives a deterministic policy.
            policy.append({"when": observed, "probability": probability, "action": choices[0],
                           "optimal_actions": choices, "expected_utility": value, "actions": branch_actions})
        policy_value = math.fsum(row["probability"] * row["expected_utility"] for row in policy if row["expected_utility"] is not None)
        # Observing more can never lower the expected utility, so a gain within
        # the tie tolerance is rounding noise and is reported as exactly zero.
        tolerance = TIE_TOLERANCE * self.utility_scale
        gain = policy_value - meu
        value_of_information = gain if gain > tolerance else 0.0
        observation_cost = math.fsum(decision.observation_costs.get(name, 0.0) for name in unknown)
        return {"decision": decision.name, "evidence": evidence, "actions": actions,
                "best_action": optimal[0], "optimal_actions": optimal, "maximum_expected_utility": meu,
                "policy": policy, "policy_expected_utility": policy_value,
                "value_of_information": value_of_information, "observation_cost": observation_cost,
                "net_value_of_information": value_of_information - observation_cost,
                "information_values": self._information_values(unknown, probabilities, branch_rows, meu),
                "unobserved_information": list(unknown)}
