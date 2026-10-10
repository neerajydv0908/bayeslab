"""Optional independent pgmpy reference, imported only when requested."""

import math

from .inference import ImpossibleEvidence
from .model import ModelError, assignments


class PgmpyReference:
    def __init__(self, network, action=None):
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
        from pgmpy.models import DiscreteBayesianNetwork

        self.network = network
        action = network.resolve_action(action)
        decision_name = network.decision.name if network.decision else None
        model = DiscreteBayesianNetwork()
        model.add_nodes_from(network.chance_names)
        for node in network.chance.values():
            # Fix decisions directly in each affected CPT. No random decision CPD.
            parents = tuple(p for p in node.parents if p != decision_name)
            model.add_edges_from((p, node.name) for p in parents)
            columns = []
            for states in assignments(parents, network.domains):
                assignment = dict(zip(parents, states))
                if decision_name:
                    assignment[decision_name] = action
                key = tuple(assignment[p] for p in node.parents)
                columns.append([node.cpt[key + (s,)] for s in node.states])
            model.add_cpds(TabularCPD(
                variable=node.name, variable_card=len(node.states), values=list(map(list, zip(*columns))),
                evidence=list(parents) if parents else None,
                evidence_card=[len(network.domains[p]) for p in parents] if parents else None,
                state_names={n: list(network.domains[n]) for n in (*parents, node.name)},
            ))
        model.check_model()
        self.model = model
        self.engine = VariableElimination(model)

    def evidence_probability(self, evidence):
        """Chain rule using only pgmpy, also detects impossible evidence."""
        observed, mass = {}, 1.0
        for name, state in evidence.items():
            factor = self.engine.query([name], evidence=observed, show_progress=False)
            mass *= float(factor.get_value(**{name: state}))
            if mass == 0:
                return 0.0
            observed[name] = state
        return mass

    def probability(self, event, evidence=None):
        evidence = {} if evidence is None else evidence
        self.network.validate_observations(event, "Event")
        self.network.validate_observations(evidence)
        if not event:
            raise ModelError("Choose at least one event state.")
        mass = self.evidence_probability(evidence)
        if mass <= 0 or not math.isfinite(mass):
            raise ImpossibleEvidence("The evidence has zero probability; the conditional result is undefined.")
        if any(name in evidence and evidence[name] != state for name, state in event.items()):
            return 0.0
        remaining = {n: s for n, s in event.items() if n not in evidence}
        if not remaining:
            return 1.0
        result = self.engine.query(list(remaining), evidence=evidence, show_progress=False)
        return float(result.get_value(**remaining))
