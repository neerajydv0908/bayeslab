"""Exact inference using factor operations and variable elimination."""

from dataclasses import dataclass
import math
import sys

from .limits import check_time
from .model import ModelError, MAX_FACTOR_ENTRIES, assignments
from .numeric import LOG10_2

# Below the smallest normal float, numbers lose precision and then become 0.
SMALLEST_NORMAL = sys.float_info.min
# Large factor products check the time limit once per this many entries (minus one, as a bit mask).
CHECK_EVERY = 0x3FFF


class ImpossibleEvidence(ModelError):
    """Conditional probabilities are undefined when evidence has probability zero."""


class FactorTooLarge(ModelError):
    """A factor would exceed MAX_FACTOR_ENTRIES, so exact elimination refuses it."""


class Underflow(ArithmeticError):
    """A product left the normal float range, so elimination is redone with ScaledFactor."""


@dataclass
class Factor:
    variables: tuple
    values: dict

    @classmethod
    def unit(cls):
        return cls((), {(): 1.0})

    def restrict(self, evidence):
        keep = tuple(name for name in self.variables if name not in evidence)
        indices = [self.variables.index(name) for name in keep]
        checks = [(i, evidence[name]) for i, name in enumerate(self.variables) if name in evidence]
        return type(self)(keep, {tuple(key[i] for i in indices): value for key, value in self.values.items()
                                 if all(key[i] == state for i, state in checks)})

    def sum_out(self, variable):
        check_time()
        index = self.variables.index(variable)
        remaining = self.variables[:index] + self.variables[index + 1:]
        buckets = {}
        for key, value in self.values.items():
            reduced = key[:index] + key[index + 1:]
            buckets.setdefault(reduced, []).append(value)
        return Factor(remaining, {key: math.fsum(values) for key, values in buckets.items()})

    def normalize(self):
        total = math.fsum(self.values.values())
        if total <= 0:
            raise ImpossibleEvidence("A zero-mass factor cannot be normalized.")
        return Factor(self.variables, {key: value / total for key, value in self.values.items()})

    def _layout(self, other, domains):
        check_time()
        variables = self.variables + tuple(name for name in other.variables if name not in self.variables)
        if math.prod(len(domains[name]) for name in variables) > MAX_FACTOR_ENTRIES:
            raise FactorTooLarge(f"An intermediate factor exceeds {MAX_FACTOR_ENTRIES:,} entries. "
                                 "Simplify this network for exact inference.")
        return variables, [variables.index(name) for name in self.variables], [variables.index(name) for name in other.variables]

    def multiply(self, other, domains):
        variables, left, right = self._layout(other, domains)
        values = {}
        for index, key in enumerate(assignments(variables, domains)):
            if not index & CHECK_EVERY:
                check_time()
            a = self.values.get(tuple(key[i] for i in left), 0)
            b = other.values.get(tuple(key[i] for i in right), 0)
            value = a * b
            if value < SMALLEST_NORMAL and a and b:
                raise Underflow
            values[key] = value
        return Factor(variables, values)


ZERO = (0.0, 0)


class ScaledFactor(Factor):
    """A factor whose values are pairs (mantissa, exponent) meaning mantissa * 2**exponent.

    Elimination uses these only when plain floats would underflow. Every value
    carries its own exponent, so values spanning far more than the float range
    stay exact up to the rounding of one multiplication or one fsum.
    """

    @classmethod
    def unit(cls):
        return cls((), {(): math.frexp(1.0)})

    @classmethod
    def of(cls, factor):
        return cls(factor.variables, {key: math.frexp(value) for key, value in factor.values.items()})

    def multiply(self, other, domains):
        variables, left, right = self._layout(other, domains)
        values = {}
        for index, key in enumerate(assignments(variables, domains)):
            if not index & CHECK_EVERY:
                check_time()
            a, e = self.values.get(tuple(key[i] for i in left), ZERO)
            b, f = other.values.get(tuple(key[i] for i in right), ZERO)
            mantissa, shift = math.frexp(a * b)
            values[key] = (mantissa, e + f + shift)
        return ScaledFactor(variables, values)

    def sum_out(self, variable):
        check_time()
        index = self.variables.index(variable)
        remaining = self.variables[:index] + self.variables[index + 1:]
        buckets = {}
        for key, pair in self.values.items():
            buckets.setdefault(key[:index] + key[index + 1:], []).append(pair)
        return ScaledFactor(remaining, {key: _add_scaled(pairs) for key, pairs in buckets.items()})

    def unscaled(self):
        """Return (factor, exponent) of plain floats with this factor = factor * 2**exponent.

        Values more than about 2**1074 times smaller than the largest become 0,
        which cannot change a normalized result.
        """
        top = max((exponent for mantissa, exponent in self.values.values() if mantissa), default=0)
        return Factor(self.variables, {key: math.ldexp(mantissa, exponent - top)
                                       for key, (mantissa, exponent) in self.values.items()}), top


def _add_scaled(pairs):
    """Sum (mantissa, exponent) pairs exactly up to one fsum rounding."""
    top = max((exponent for mantissa, exponent in pairs if mantissa), default=None)
    if top is None:
        return ZERO
    mantissa, shift = math.frexp(math.fsum(math.ldexp(m, e - top) for m, e in pairs))
    return mantissa, top + shift


def multiply_all(factors, domains):
    result = (type(factors[0]) if factors else Factor).unit()
    for factor in factors:
        result = result.multiply(factor, domains)
    return result


class Inference:
    def __init__(self, network):
        self.network = network

    def _relevant(self, names):
        """The given chance nodes together with all of their chance-node ancestors.

        Every other chance node is barren for the query: it is neither queried
        nor observed and has no queried or observed descendant, so summing it
        out multiplies the answer by exactly 1. The decision is fixed by the
        selected action, so the search stops there and ignores information arcs.
        """
        chance = self.network.chance
        relevant, pending = set(), [name for name in names if name in chance]
        while pending:
            name = pending.pop()
            if name not in relevant:
                relevant.add(name)
                pending.extend(parent for parent in chance[name].parents if parent in chance)
        return relevant

    def _factor(self, targets, evidence, action, trace=None):
        """Return (factor, exponent): P(targets, evidence | do(action)) = factor * 2**exponent.

        Elimination runs on plain floats, and then the exponent is 0. If any
        product would underflow (only for evidence around 1e-308 or less likely),
        the elimination is redone with ScaledFactor so the answer stays accurate.
        """
        network = self.network
        action = network.resolve_action(action)
        fixed = dict(evidence)
        if network.decision:
            fixed[network.decision.name] = action
        relevant = self._relevant((*targets, *evidence))
        factors = [Factor(node.parents + (node.name,), node.cpt).restrict(fixed)
                   for name, node in network.chance.items() if name in relevant]
        remaining = relevant - set(targets) - set(fixed)
        if trace is not None:
            pruned = [name for name in network.chance_names if name not in relevant]
            if pruned:
                trace.append({"operation": "prune", "variables": pruned, "description":
                              f"Skip {', '.join(pruned)}: not queried, not observed, and not an ancestor of anything "
                              f"queried or observed, so summing {'it' if len(pruned) == 1 else 'them'} out would only multiply by 1."})
            trace.append({"operation": "condition", "description": f"Start with {len(factors)} CPT factors; restrict observed states and the selected action."})
        steps = []
        try:
            result, exponent = self._eliminate(factors, remaining, steps), 0
        except Underflow:
            steps = [{"operation": "rescale", "description":
                      "Some products fell below the smallest float, so the elimination was redone with every "
                      "value stored as a mantissa and its own power-of-two exponent."}]
            result, exponent = self._eliminate([ScaledFactor.of(f) for f in factors], remaining, steps).unscaled()
        if trace is not None:
            trace.extend(steps)
        return result, exponent

    def _eliminate(self, factors, remaining, steps):
        network = self.network
        remaining = set(remaining)
        while remaining:
            # Prefer the smallest resulting product, with stable tie-breaking.
            def cost(name):
                scope = set().union(*(set(f.variables) for f in factors if name in f.variables))
                return math.prod(len(network.domains[v]) for v in scope), name
            variable = min(remaining, key=cost)
            involved = [factor for factor in factors if variable in factor.variables]
            factors = [factor for factor in factors if variable not in factor.variables]
            reduced = multiply_all(involved, network.domains).sum_out(variable)
            factors.append(reduced)
            remaining.remove(variable)
            steps.append({"operation": "eliminate", "variable": variable, "scope": list(reduced.variables),
                          "entries": len(reduced.values), "description": f"Multiply {len(involved)} factors and sum out {variable}."})
        return multiply_all(factors, network.domains)

    def _distribution(self, targets, evidence, action, trace=None):
        """Return (distribution, (total, exponent)), where P(evidence) = total * 2**exponent."""
        action = self.network.resolve_action(action)
        if math.prod(len(self.network.domains[name]) for name in targets) > MAX_FACTOR_ENTRIES:
            raise FactorTooLarge(f"The requested distribution exceeds {MAX_FACTOR_ENTRIES:,} entries.")
        factor, exponent = self._factor(targets, evidence, action, trace)
        total = math.fsum(factor.values.values())
        if total <= 0:
            raise ImpossibleEvidence("The evidence has zero probability (or underflows floating-point precision); the conditional result is undefined.")
        fixed = dict(evidence)
        if self.network.decision:
            fixed[self.network.decision.name] = action
        values = {}
        for key in assignments(targets, self.network.domains):
            chosen = dict(zip(targets, key))
            if any(chosen[name] != fixed[name] for name in targets if name in fixed):
                values[key] = 0.0
            else:
                values[key] = factor.values.get(tuple(chosen[name] for name in factor.variables), 0.0) / total
        if trace is not None:
            trace.append({"operation": "normalize", "evidence_probability": math.ldexp(total, exponent),
                          "description": "Normalize the remaining factor by the probability of the evidence."})
        return Factor(tuple(targets), values), (total, exponent)

    def probability(self, event, evidence=None, action=None):
        return self.query(event, evidence, action)["probability"]

    def event_probability(self, event, action=None):
        self.network.validate_observations(event, "Event")
        if not event:
            self.network.resolve_action(action)
            return 1.0
        return self.probability(event, action=action)

    def conditional_probability(self, event, evidence, action=None):
        return self.probability(event, evidence, action)

    def joint_probability(self, world, action=None):
        self.network.validate_observations(world, "Joint assignment")
        if set(world) != set(self.network.chance_names):
            raise ModelError("A complete joint assignment must specify every chance variable.")
        return self.event_probability(world, action)

    def distribution(self, target, evidence=None, action=None):
        evidence = {} if evidence is None else evidence
        self.network.validate_observations(evidence)
        if target not in self.network.chance_names:
            raise ModelError(f"Unknown chance node: {target}.")
        factor, _ = self._distribution((target,), evidence, action)
        return {key[0]: value for key, value in factor.values.items()}

    def query(self, event, evidence=None, action=None):
        evidence = {} if evidence is None else evidence
        self.network.validate_observations(event, "Event")
        self.network.validate_observations(evidence)
        if not event:
            raise ModelError("Choose at least one event state.")
        trace = []
        # Only keep a full distribution for a single target. For a larger joint
        # event, two scalar eliminations avoid materializing every target state.
        if len(event) == 1:
            targets = tuple(event)
            factor, (mass, exponent) = self._distribution(targets, evidence, action, trace)
            probability = factor.values[tuple(event.values())]
            distribution = [{"state": key[0], "probability": value} for key, value in factor.values.items()]
        else:
            scalar, exponent = self._factor((), evidence, action)
            mass = scalar.values.get((), 0)
            if mass <= 0:
                raise ImpossibleEvidence("The evidence has zero probability (or underflows floating-point precision); the conditional result is undefined.")
            if any(name in evidence and evidence[name] != state for name, state in event.items()):
                probability = 0.0  # the event contradicts the evidence
            else:
                joint, joint_exponent = self._factor((), {**evidence, **event}, action, trace)
                probability = math.ldexp(joint.values.get((), 0) / mass, joint_exponent - exponent)
            distribution = []
            trace.append({"operation": "normalize", "evidence_probability": math.ldexp(mass, exponent),
                          "description": "Divide P(event, evidence) by P(evidence)."})
        return {"probability": min(1.0, max(0.0, probability)), "event": event, "evidence": evidence, "action": action,
                "evidence_probability": math.ldexp(mass, exponent),
                "log10_evidence_probability": math.log10(mass) + exponent * LOG10_2,
                "distribution": distribution, "trace": trace, "method": "Exact variable elimination"}
