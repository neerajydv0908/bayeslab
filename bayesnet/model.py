"""Validated, finite discrete network models. No third-party dependencies."""

from dataclasses import dataclass, field
from itertools import product
import math
from operator import itemgetter


class ModelError(ValueError):
    """The network or query is not well defined."""


MAX_FACTOR_ENTRIES = 200_000
# Expected utilities, their differences and the sensitivity lines add up values
# from every utility table, so the largest possible total must stay far from
# the floating-point limit (about 1.8e308).
MAX_TOTAL_UTILITY = 1e300


def assignments(variables, domains):
    return product(*(domains[name] for name in variables))


def values_of(names):
    """A fast function mapping a world (dict) to the tuple of its values for `names`."""
    if not names:
        return lambda world: ()
    if len(names) == 1:
        name = names[0]
        return lambda world: (world[name],)
    return itemgetter(*names)


def number(value, location):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ModelError(f"{location} must be a finite number.")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ModelError(f"{location} must be a finite number.") from None
    if not math.isfinite(result):
        raise ModelError(f"{location} must be a finite number.")
    return result


def string(value, location):
    if not isinstance(value, str) or not value.strip() or len(value) > 80:
        raise ModelError(f"{location} must be a nonempty string of at most 80 characters.")
    return value


def string_list(value, location):
    if not isinstance(value, list):
        raise ModelError(f"{location} must be a list.")
    values = tuple(string(item, location) for item in value)
    if len(set(values)) != len(values):
        raise ModelError(f"{location} contains duplicates.")
    return values


@dataclass(frozen=True)
class ChanceNode:
    name: str
    states: tuple
    parents: tuple
    cpt: dict


@dataclass(frozen=True)
class DecisionNode:
    name: str
    states: tuple
    information: tuple
    # Utility units paid to observe each information variable (default: free).
    observation_costs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class UtilityNode:
    name: str
    parents: tuple
    table: dict


class Network:
    """A Bayesian network, optionally with one decision and additive utilities.

    Information arcs indicate observations available before the decision. They
    participate in cycle validation but do not define a random decision CPT.
    """

    def __init__(self, data):
        if not isinstance(data, dict):
            raise ModelError("A model must be a JSON object.")
        self.title = string(data.get("name", "Untitled network"), "Model name")
        self.description = data.get("description", "")
        if not isinstance(self.description, str) or len(self.description) > 4000:
            raise ModelError("Description must be text of at most 4,000 characters.")
        raw_chance = data.get("chance_nodes", [])
        raw_decisions = data.get("decision_nodes", [])
        raw_utilities = data.get("utility_nodes", [])
        for key, values in (("chance_nodes", raw_chance), ("decision_nodes", raw_decisions), ("utility_nodes", raw_utilities)):
            if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
                raise ModelError(f"{key} must be a list of objects.")
        if not raw_chance:
            raise ModelError("Add at least one chance node.")
        if len(raw_chance) + len(raw_decisions) + len(raw_utilities) > 30:
            raise ModelError("This exact inference engine supports at most 30 nodes.")
        if len(raw_decisions) > 1:
            raise ModelError("This version supports one decision node per network.")
        if raw_decisions and not raw_utilities:
            raise ModelError("A decision network needs at least one utility node.")
        self.domains = {}
        names = []
        for node in raw_chance + raw_decisions + raw_utilities:
            name = string(node.get("name"), "Node name")
            if name in names:
                raise ModelError(f"Duplicate node name: {name}.")
            names.append(name)
        for node in raw_chance + raw_decisions:
            name = node["name"]
            states = string_list(node.get("states"), f"States of {name}")
            if not 2 <= len(states) <= 8:
                raise ModelError(f"{name} must have between 2 and 8 states.")
            self.domains[name] = states
        self.chance_names = tuple(node["name"] for node in raw_chance)
        self.decision = None
        if raw_decisions:
            raw = raw_decisions[0]
            info = string_list(raw.get("information", []), f"Information of {raw['name']}")
            if any(name not in self.chance_names for name in info):
                raise ModelError("Decision information must refer to chance nodes.")
            self.decision = DecisionNode(raw["name"], self.domains[raw["name"]], info,
                                         self._observation_costs(raw, info))
        self.chance = {}
        for raw in raw_chance:
            name = raw["name"]
            parents = self._parents(raw, name)
            rows = self._rows(raw.get("cpt"), parents, f"CPT of {name}")
            states = self.domains[name]
            cpt = {}
            for key, row in rows.items():
                condition = ", ".join(f"{parent}={value}" for parent, value in zip(parents, key))
                given = f" given {condition}" if condition else ""
                probabilities = row.get("probabilities")
                if not isinstance(probabilities, dict):
                    raise ModelError(f"The CPT row of {name}{given} needs a 'probabilities' object.")
                missing = [state for state in states if state not in probabilities]
                unknown = [str(state) for state in probabilities if state not in states]
                if missing or unknown:
                    problems = ([f"missing {', '.join(missing)}"] if missing else []) + (
                        [f"unknown {', '.join(unknown)}"] if unknown else [])
                    raise ModelError(f"The CPT row of {name}{given} must list exactly the states "
                                     f"{', '.join(states)} ({'; '.join(problems)}).")
                for state in states:
                    label = f"P({name}={state} | {condition})" if condition else f"P({name}={state})"
                    probability = number(probabilities[state], label)
                    if not 0 <= probability <= 1:
                        raise ModelError(f"{label} is {probability:g}, but probabilities must be between 0 and 1.")
                    cpt[key + (state,)] = probability
                total = math.fsum(cpt[key + (state,)] for state in states)
                if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-9):
                    raise ModelError(f"The probabilities of {name}{given} add up to {total:.10g}, "
                                     "but each CPT row must add up to 1.")
            self.chance[name] = ChanceNode(name, self.domains[name], parents, cpt)
        self.utilities = []
        for raw in raw_utilities:
            name = raw["name"]
            parents = self._parents(raw, name)
            rows = self._rows(raw.get("table"), parents, f"Utility table of {name}")
            self.utilities.append(UtilityNode(name, parents, {key: number(row.get("value"), f"Utility in {name}") for key, row in rows.items()}))
        # Largest absolute total utility any outcome can produce. It also sets the
        # scale for tie detection in decision analysis.
        self.utility_scale = sum(max(abs(value) for value in node.table.values()) for node in self.utilities)
        if self.utility_scale > MAX_TOTAL_UTILITY:
            raise ModelError("Utility values are too large to calculate with safely: the largest possible total "
                             f"utility must be at most {MAX_TOTAL_UTILITY:g}. Measure utilities in larger units, "
                             "such as thousands or millions.")
        parent_map = {name: node.parents for name, node in self.chance.items()}
        if self.decision:
            parent_map[self.decision.name] = self.decision.information
        self.order = self._topological(parent_map)
        self.source = data

    @staticmethod
    def _observation_costs(raw, information):
        costs = raw.get("observation_costs", {})
        if not isinstance(costs, dict):
            raise ModelError(f"Observation costs of {raw['name']} must be an object mapping variables to costs.")
        result = {}
        for name, value in costs.items():
            if name not in information:
                observed = ", ".join(information) or "nothing"
                raise ModelError(f"{raw['name']} has an observation cost for {name}, but {name} is not in its "
                                 f"information set (observed before deciding: {observed}).")
            cost = number(value, f"The cost of observing {name}")
            if not 0 <= cost <= MAX_TOTAL_UTILITY:
                raise ModelError(f"The cost of observing {name} is {cost:g}; costs must be zero or positive "
                                 f"and at most {MAX_TOTAL_UTILITY:g}.")
            result[name] = cost
        return result

    def _parents(self, raw, name):
        parents = string_list(raw.get("parents", []), f"Parents of {name}")
        for parent in parents:
            if parent == name or parent not in self.domains:
                raise ModelError(f"Invalid parent {parent} in {name}.")
        return parents

    def _rows(self, rows, parents, location):
        size = math.prod(len(self.domains[name]) for name in parents)
        if size > MAX_FACTOR_ENTRIES // 8:
            raise ModelError(f"{location} is too large for interactive exact inference.")
        if not isinstance(rows, list) or len(rows) != size:
            raise ModelError(f"{location} needs exactly {size} rows, one per parent assignment.")
        result = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("when"), dict) or set(row["when"]) != set(parents):
                expected = ", ".join(parents) if parents else "no parents, so use {}"
                raise ModelError(f"Each 'when' in {location} must name exactly its parents ({expected}).")
            when = row["when"]
            for name, value in when.items():
                if value not in self.domains[name]:
                    raise ModelError(f"{location} uses {value!r} for {name}, which is not one of its states "
                                     f"({', '.join(self.domains[name])}).")
            key = tuple(when[name] for name in parents)
            if key in result:
                raise ModelError(f"Duplicate parent assignment in {location}.")
            result[key] = row
        return result

    @staticmethod
    def _topological(parents):
        pending = dict(parents)
        result = []
        while pending:
            ready = [name for name, dependencies in pending.items() if all(parent in result for parent in dependencies)]
            if not ready:
                raise ModelError("The network contains a directed cycle, including decision information arcs.")
            for name in ready:
                result.append(name)
                del pending[name]
        return tuple(result)

    def validate_observations(self, observations, location="Evidence", allowed=None):
        if not isinstance(observations, dict):
            raise ModelError(f"{location} must be an object mapping node names to states.")
        allowed = self.chance_names if allowed is None else allowed
        for name, value in observations.items():
            if name not in allowed:
                if self.decision and name == self.decision.name:
                    raise ModelError(f"{location}: {name} is the decision, not a chance variable. "
                                     "Choose its action separately.")
                if name in self.chance:
                    raise ModelError(f"{location}: {name} cannot be used here.")
                raise ModelError(f"{location}: {name} is not a chance variable in this model.")
            if value not in self.domains[name]:
                raise ModelError(f"{location}: {value!r} is not a state of {name} "
                                 f"(states: {', '.join(self.domains[name])}).")

    def validate_action(self, action):
        if self.decision:
            if action not in self.decision.states:
                raise ModelError(f"Select an action for {self.decision.name}: {', '.join(self.decision.states)}.")
        elif action is not None:
            raise ModelError("This Bayesian network has no decision node.")

    def resolve_action(self, action):
        """A probability query needs an action only if it changes chance CPTs."""
        if action is None and self.decision and not any(
            self.decision.name in node.parents for node in self.chance.values()
        ):
            return self.decision.states[0]
        self.validate_action(action)
        return action

    def summary(self):
        edges = sum(len(n.parents) for n in self.chance.values()) + sum(len(u.parents) for u in self.utilities)
        if self.decision:
            edges += len(self.decision.information)
        return {"name": self.title, "chance_nodes": len(self.chance), "decision_nodes": int(self.decision is not None),
                "utility_nodes": len(self.utilities), "edges": edges, "order": list(self.order),
                "joint_assignments": math.prod(len(self.domains[name]) for name in self.chance_names)}
