"""Exact inference by visiting complete possible worlds, depth first.

This baseline intentionally does not use the factor or elimination code. It
assigns the chance variables one at a time in topological order, as in the
textbook ENUMERATION-ASK, so each CPT entry is multiplied in once and shared by
every world below it. Evidence variables only take their observed state, and a
branch is skipped as soon as its probability is zero, because every world below
it is zero too. Tiny products are rescaled by exact powers of two, so no world
probability ever underflows.
"""

import math

from .inference import ImpossibleEvidence
from .limits import check_time
from .model import ModelError, assignments, values_of
from .numeric import RESCALE_BELOW, ScaledSum, describe

MAX_WORLDS = 1_048_576


class Enumeration:
    def __init__(self, network):
        self.network = network
        # Topological order: every chance parent is assigned before its child.
        self.order = tuple(name for name in network.order if name in network.chance)

    def _check_size(self, evidence=None):
        """Return the number of possible worlds; raise if too many agree with the evidence.

        Observed variables take a single state, so the limit applies to the
        worlds that agree with the evidence, which are the only ones visited.
        """
        domains, names = self.network.domains, self.network.chance_names
        total = math.prod(len(domains[n]) for n in names)
        if evidence is None:
            if total > MAX_WORLDS:
                raise ModelError(f"Enumeration exceeds {MAX_WORLDS:,} worlds ({total:,} in this network). "
                                 "Use variable elimination.")
            return total
        visited = math.prod(1 if n in evidence else len(domains[n]) for n in names)
        if visited > MAX_WORLDS:
            raise ModelError(f"Enumeration exceeds {MAX_WORLDS:,} worlds: {visited:,} worlds agree with this "
                             "evidence. Observe more variables or use variable elimination.")
        return total

    def _joint(self, world, action):
        fixed = dict(world)
        if self.network.decision:
            fixed[self.network.decision.name] = action
        return math.prod(node.cpt[tuple(fixed[p] for p in node.parents) + (fixed[node.name],)]
                         for node in self.network.chance.values())

    def _walk(self, evidence, action, leaf):
        """Visit, depth first, each world that agrees with the evidence and has nonzero probability.

        Calls leaf(world, value, exponent) once per world, whose probability is
        value * 2**exponent. `world` is one dict reused between calls.
        """
        network = self.network
        world = {network.decision.name: action} if network.decision else {}
        steps = []
        for name in self.order:
            node = network.chance[name]
            states = (evidence[name],) if name in evidence else node.states
            steps.append((name, values_of(node.parents), node.cpt, states))
        last = len(steps) - 1
        # Check the time limit a few levels above the leaves: often enough to stop
        # promptly, rarely enough to cost nothing measurable.
        check_depth = max(0, last - 3)

        def visit(depth, probability, exponent):
            name, parent_values, cpt, states = steps[depth]
            if depth == check_depth:
                check_time()
            prefix = parent_values(world)
            for state in states:
                entry = cpt[prefix + (state,)]
                if entry == 0:
                    continue  # every world below this branch has probability zero
                shift = exponent
                if entry < RESCALE_BELOW:
                    entry, extra = math.frexp(entry)
                    shift += extra
                value = probability * entry
                if value < RESCALE_BELOW:
                    value, extra = math.frexp(value)
                    shift += extra
                world[name] = state
                if depth == last:
                    leaf(world, value, shift)
                else:
                    visit(depth + 1, value, shift)

        visit(0, 1.0, 0)

    def _sum(self, evidence, action):
        """P(evidence | do(action)) as a ScaledSum over the worlds that agree with it."""
        total = ScaledSum()
        self._walk(evidence, action, lambda world, value, exponent: total.add(value, exponent))
        return total

    def joint_probability(self, world, action=None):
        self.network.validate_observations(world, "Joint assignment")
        if set(world) != set(self.network.chance_names):
            raise ModelError("A complete joint assignment must specify every chance variable.")
        return self._joint(world, self.network.resolve_action(action))

    def worlds(self, action=None):
        """Every world with its probability, including zero-probability worlds (for joint tables)."""
        self._check_size()
        action = self.network.resolve_action(action)
        for values in assignments(self.network.chance_names, self.network.domains):
            world = dict(zip(self.network.chance_names, values))
            yield {"assignment": world, "probability": self._joint(world, action)}

    def event_probability(self, event, action=None):
        self.network.validate_observations(event, "Event")
        self._check_size(event)
        return self._sum(event, self.network.resolve_action(action)).value()

    def probability(self, event, evidence=None, action=None):
        return self.query(event, evidence, action)["probability"]

    def conditional_probability(self, event, evidence, action=None):
        return self.probability(event, evidence, action)

    def distribution(self, target, evidence=None, action=None):
        if target not in self.network.chance_names:
            raise ModelError(f"Unknown chance node: {target}.")
        result = self.query({target: self.network.domains[target][0]}, evidence, action)
        return {row["state"]: row["probability"] for row in result["distribution"]}

    def query(self, event, evidence=None, action=None):
        evidence = {} if evidence is None else evidence
        self.network.validate_observations(event, "Event")
        self.network.validate_observations(evidence)
        if not event:
            raise ModelError("Choose at least one event state.")
        size = self._check_size(evidence)
        resolved = self.network.resolve_action(action)
        target = next(iter(event)) if len(event) == 1 else None
        mass = ScaledSum()
        groups = {state: ScaledSum() for state in self.network.domains[target]} if target else {}

        def leaf(world, value, exponent):
            mass.add(value, exponent)
            if target:
                groups[world[target]].add(value, exponent)

        self._walk(evidence, resolved, leaf)
        if mass.is_zero():
            raise ImpossibleEvidence("The evidence has zero probability, so the conditional result is undefined.")
        if target:
            joint = groups[event[target]]
        elif any(name in evidence and evidence[name] != state for name, state in event.items()):
            joint = ScaledSum()  # the event contradicts the evidence
        else:
            joint = self._sum({**evidence, **event}, resolved)
        evidence_log10 = mass.log10()
        evidence_text = describe(mass.value(), evidence_log10)
        joint_text = describe(joint.value(), None if joint.is_zero() else joint.log10())
        trace = [{"operation": "enumerate", "description":
                  f"Walk depth first through the {mass.count:,} worlds that agree with the evidence and have "
                  f"nonzero probability (of {size:,} possible), multiplying in one CPT entry per step."}]
        if mass.rescaled or joint.rescaled:
            trace.append({"operation": "rescale", "description":
                          "Some products fell below 2^-500, so they were rescaled by exact powers of two "
                          "instead of underflowing."})
        trace += [
            {"operation": "filter", "description": f"Add up those worlds: P(evidence) = {evidence_text}."},
            {"operation": "sum", "description":
             f"Add up the {joint.count:,} worlds where the event also holds: P(event, evidence) = {joint_text}."},
            {"operation": "normalize", "description": f"Divide {joint_text} by {evidence_text}."}]
        return {"probability": min(1.0, max(0.0, joint.ratio(mass))), "event": event, "evidence": evidence,
                "action": action, "evidence_probability": mass.value(), "log10_evidence_probability": evidence_log10,
                "distribution": [{"state": state, "probability": min(1.0, max(0.0, group.ratio(mass)))}
                                 for state, group in groups.items()],
                "trace": trace, "method": "Exact enumeration"}
