"""Compare BayesLab with an exact-fraction oracle on random networks.

Run from the project root:
    python3 -m experiments.random_oracle --networks 1500

Builds random networks (2-3 states, up to two parents, zero probabilities,
decisions that change chance CPTs, several utility nodes, observation costs)
and checks both inference engines and both decision methods against brute
force over every world in exact rational arithmetic: probabilities, impossible
evidence, expected utilities, optimal actions and ties, policies, the value of
information (joint and per variable, net of costs), and the sensitivity
envelope for 2-6 actions. The oracle shares no code with the engines.
"""
import argparse
import itertools
import random
import traceback
from fractions import Fraction as F

from bayesnet import Network, Inference, Enumeration, DecisionAnalysis, ImpossibleEvidence

TOL = 1e-9


def rand_dist(rng, k, zeros):
    w = [0 if (zeros and rng.random() < 0.25) else rng.randint(1, 20) for _ in range(k)]
    if sum(w) == 0:
        w[rng.randrange(k)] = 1
    s = sum(w)
    # Use binary-exact-ish values: round to 1/2**10 grid so floats sum exactly to 1
    vals = [round(x / s * 1024) for x in w]
    diff = 1024 - sum(vals)
    i = max(range(k), key=lambda j: vals[j])
    vals[i] += diff
    if vals[i] < 0:
        vals = [0] * k; vals[0] = 1024
    return [v / 1024 for v in vals]


def random_model(rng, with_decision):
    n = rng.randint(1, 6)
    names = [f"V{i}" for i in range(n)]
    states = {nm: [f"s{j}" for j in range(rng.randint(2, 3))] for nm in names}
    zeros = rng.random() < 0.4
    dname = "D"
    dstates = [f"a{j}" for j in range(rng.randint(2, 3))]
    k = rng.randint(0, n)  # first k nodes may be information for D
    info = [nm for nm in names[:k] if rng.random() < 0.6] if with_decision else []
    chance = []
    for i, nm in enumerate(names):
        cands = names[:i]
        parents = rng.sample(cands, min(len(cands), rng.randint(0, 2)))
        if with_decision and i >= k and rng.random() < 0.4:
            parents.append(dname)
        pdom = [dstates if p == dname else states[p] for p in parents]
        cpt = []
        for combo in itertools.product(*pdom):
            probs = rand_dist(rng, len(states[nm]), zeros)
            cpt.append({"when": dict(zip(parents, combo)), "probabilities": dict(zip(states[nm], probs))})
        chance.append({"name": nm, "states": states[nm], "parents": parents, "cpt": cpt})
    model = {"name": "fuzz", "chance_nodes": chance}
    if with_decision:
        model["decision_nodes"] = [{"name": dname, "states": dstates, "information": info}]
        if info and rng.random() < 0.5:
            model["decision_nodes"][0]["observation_costs"] = {n: rng.choice([0, 0.5, 1, 3.25]) for n in info if rng.random() < 0.7}
        utils = []
        for u in range(rng.randint(1, 3)):
            pool = names + [dname]
            parents = rng.sample(pool, min(len(pool), rng.randint(1, 3)))
            pdom = [dstates if p == dname else states[p] for p in parents]
            table = [{"when": dict(zip(parents, combo)), "value": rng.choice([rng.randint(-20, 20), round(rng.uniform(-50, 50), 3)])}
                     for combo in itertools.product(*pdom)]
            utils.append({"name": f"U{u}", "parents": parents, "table": table})
        model["utility_nodes"] = utils
    return model


class Oracle:
    def __init__(self, model):
        self.m = model
        self.chance = model["chance_nodes"]
        self.names = [c["name"] for c in self.chance]
        self.dom = {c["name"]: c["states"] for c in self.chance}
        self.dec = model.get("decision_nodes", [None])[0] if model.get("decision_nodes") else None
        if self.dec:
            self.dom[self.dec["name"]] = self.dec["states"]

    def worlds(self, action):
        for combo in itertools.product(*(self.dom[n] for n in self.names)):
            w = dict(zip(self.names, combo))
            if self.dec:
                w[self.dec["name"]] = action
            p = F(1)
            for c in self.chance:
                row = next(r for r in c["cpt"] if all(w[k] == v for k, v in r["when"].items()))
                p *= F(row["probabilities"][w[c["name"]]])
            yield w, p

    def prob(self, event, evidence, action):
        num = den = F(0)
        for w, p in self.worlds(action):
            if all(w[k] == v for k, v in evidence.items()):
                den += p
                if all(w[k] == v for k, v in event.items()):
                    num += p
        return None if den == 0 else num / den

    def utility(self, w):
        tot = F(0)
        for u in self.m["utility_nodes"]:
            row = next(r for r in u["table"] if all(w[k] == v for k, v in r["when"].items()))
            tot += F(row["value"])
        return tot

    def eu(self, action, evidence):
        num = den = F(0)
        for w, p in self.worlds(action):
            if all(w[k] == v for k, v in evidence.items()):
                den += p
                num += p * self.utility(w)
        return None if den == 0 else num / den

    def analyze(self, evidence):
        acts = self.dec["states"]
        eus = {a: self.eu(a, evidence) for a in acts}
        if any(v is None for v in eus.values()):
            return None
        meu = max(eus.values())
        unknown = [n for n in self.dec["information"] if n not in evidence]
        policy = []
        pev = sum(p for w, p in self.worlds(acts[0]) if all(w[k] == v for k, v in evidence.items()))
        for combo in itertools.product(*(self.dom[n] for n in unknown)):
            obs = {**evidence, **dict(zip(unknown, combo))}
            pobs = sum(p for w, p in self.worlds(acts[0]) if all(w[k] == v for k, v in obs.items())) / pev
            if pobs == 0:
                policy.append((obs, F(0), None, None))
                continue
            beus = {a: self.eu(a, obs) for a in acts}
            best = max(beus.values())
            policy.append((obs, pobs, best, [a for a in acts if beus[a] == best]))
        pol_eu = sum(p * b for _, p, b, _ in policy if b is not None)
        per_var = {}
        for x in unknown:
            total = F(0)
            for s_ in self.dom[x]:
                obs = {**evidence, x: s_}
                px = sum(p for w, p in self.worlds(acts[0]) if all(w[k] == v for k, v in obs.items())) / pev
                if px == 0:
                    continue
                total += px * max(self.eu(a, obs) for a in acts)
            per_var[x] = total - meu
        return {"per_var": per_var,"eus": eus, "meu": meu, "optimal": [a for a in acts if eus[a] == meu], "policy": policy,
                "policy_eu": pol_eu, "voi": pol_eu - meu}


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def check(seed, stats):
    rng = random.Random(seed)
    with_decision = rng.random() < 0.6
    model = random_model(rng, with_decision)
    net = Network(model)
    orc = Oracle(model)
    acts = model["decision_nodes"][0]["states"] if with_decision else [None]
    action_matters = with_decision and any("D" in c["parents"] for c in model["chance_nodes"])
    engines = [Enumeration(net), Inference(net)]
    # Probability queries
    for _ in range(12):
        names = list(orc.names)
        rng.shuffle(names)
        ev_n = rng.randint(1, len(names))
        event = {n: rng.choice(orc.dom[n]) for n in names[:ev_n]}
        evidence = {n: rng.choice(orc.dom[n]) for n in names[ev_n:] if rng.random() < 0.7}
        if rng.random() < 0.15 and names:  # overlapping event/evidence
            n = rng.choice(names); evidence[n] = rng.choice(orc.dom[n]); event[n] = rng.choice(orc.dom[n])
        action = rng.choice(acts) if action_matters else None
        expected = orc.prob(event, evidence, action if action_matters else acts[0])
        for eng in engines:
            try:
                got = eng.probability(event, evidence, action)
            except ImpossibleEvidence:
                got = None
            stats["prob"] += 1
            if (got is None) != (expected is None) or (got is not None and not close(got, expected)):
                return f"seed {seed}: {type(eng).__name__} P({event}|{evidence}, a={action}) = {got}, oracle {float(expected) if expected is not None else None}"
        if len(event) == 1 and expected is not None:
            (t, s), = event.items()
            for eng in engines:
                d = eng.distribution(t, evidence, action)
                if not close(sum(d.values()), 1):
                    return f"seed {seed}: distribution not normalized {d}"
    if not with_decision:
        return None
    # Decision analysis
    info = model["decision_nodes"][0]["information"]
    for _ in range(4):
        evidence = {n: rng.choice(orc.dom[n]) for n in info if rng.random() < 0.4}
        exp = orc.analyze(evidence)
        for method in ("enumeration", "elimination"):
            da = DecisionAnalysis(net, method)
            try:
                res = da.analyze(evidence)
            except ImpossibleEvidence:
                res = None
            stats["decision"] += 1
            if (res is None) != (exp is None):
                return f"seed {seed}: {method} impossible mismatch evidence={evidence}: got {res is None}, oracle {exp is None}"
            if res is None:
                continue
            for row in res["actions"]:
                if not close(row["expected_utility"], exp["eus"][row["action"]]):
                    return f"seed {seed}: {method} EU({row['action']}|{evidence}) {row['expected_utility']} vs {float(exp['eus'][row['action']])}"
            if not close(res["maximum_expected_utility"], exp["meu"]):
                return f"seed {seed}: {method} MEU mismatch"
            if res["optimal_actions"] != exp["optimal"]:
                return f"seed {seed}: {method} optimal {res['optimal_actions']} vs {exp['optimal']} (eus {[float(v) for v in exp['eus'].values()]})"
            if not close(res["policy_expected_utility"], exp["policy_eu"]):
                return f"seed {seed}: {method} policy EU {res['policy_expected_utility']} vs {float(exp['policy_eu'])}"
            if not close(res["value_of_information"], exp["voi"], 1e-8):
                return f"seed {seed}: {method} VOI {res['value_of_information']} vs {float(exp['voi'])}"
            costs = model["decision_nodes"][0].get("observation_costs", {})
            unknown = [n for n in info if n not in evidence]
            if [r["variable"] for r in res["information_values"]] != unknown:
                return f"seed {seed}: {method} information_values variables {res['information_values']} vs {unknown}"
            for r in res["information_values"]:
                ev = exp["per_var"][r["variable"]]
                if ev < 0:
                    return f"seed {seed}: oracle per-variable VOI negative"
                if not close(r["value"], ev, 1e-8):
                    return f"seed {seed}: {method} VOI({r['variable']}) {r['value']} vs {float(ev)}"
                c = costs.get(r["variable"], 0)
                if r["cost"] != c or not close(r["net_value"], ev - F(c), 1e-8):
                    return f"seed {seed}: {method} net VOI({r['variable']}) wrong"
                stats["pervar>0"] += r["value"] > 0
            tc = sum(F(costs.get(n, 0)) for n in unknown)
            if not close(res["observation_cost"], tc) or not close(res["net_value_of_information"], exp["voi"] - tc, 1e-8):
                return f"seed {seed}: {method} joint net VOI wrong"
            if exp["voi"] < 0:
                return f"seed {seed}: oracle VOI negative?!"
            for got, (obs, p, best, opt) in zip(res["policy"], exp["policy"]):
                if got["when"] != obs or not close(got["probability"], p):
                    return f"seed {seed}: {method} policy row {got['when']} p={got['probability']} vs {obs} {float(p)}"
                if best is None:
                    if got["action"] is not None:
                        return f"seed {seed}: {method} unreachable branch given an action"
                    continue
                if not close(got["expected_utility"], best) or got["optimal_actions"] != opt:
                    return f"seed {seed}: {method} branch {obs}: {got['optimal_actions']} {got['expected_utility']} vs {opt} {float(best)}"
            stats["voi>0"] += res["value_of_information"] > 0
        # single EU
        for a in acts:
            for method in ("enumeration", "elimination"):
                try:
                    v = DecisionAnalysis(net, method).expected_utility(a, evidence)["expected_utility"]
                except ImpossibleEvidence:
                    v = None
                e = orc.eu(a, evidence)
                if (v is None) != (e is None) or (v is not None and not close(v, e)):
                    return f"seed {seed}: {method} expected_utility({a}) {v} vs {e}"
    return None


def check_sensitivity(seed, stats):
    rng = random.Random(10_000 + seed)
    nact = rng.randint(2, 6)
    acts = [f"a{j}" for j in range(nact)]
    pr = rng.choice([0.1, 0.25, 0.5, 0.75])
    model = {"name": "sens", "chance_nodes": [
        {"name": "T", "states": ["hi", "lo"], "parents": [], "cpt": [{"when": {}, "probabilities": {"hi": pr, "lo": 1 - pr}}]},
        {"name": "O", "states": ["x", "y"], "parents": ["T"], "cpt": [
            {"when": {"T": "hi"}, "probabilities": {"x": 0.75, "y": 0.25}}, {"when": {"T": "lo"}, "probabilities": {"x": 0.25, "y": 0.75}}]}],
        "decision_nodes": [{"name": "D", "states": acts, "information": ["O"]}], "utility_nodes": []}
    for u in range(rng.randint(1, 2)):
        parents = rng.choice([["T", "D"], ["D"], ["T"], ["D", "T"]])
        table = [{"when": dict(zip(parents, c)), "value": rng.choice([rng.randint(-10, 10), rng.randint(-3, 3)])}
                 for c in itertools.product(*([acts if p == "D" else ["hi", "lo"]] [0] for p in parents))]
        model["utility_nodes"].append({"name": f"U{u}", "parents": parents, "table": table})
    if rng.random() < 0.2:  # duplicate an action's utilities to force exact ties
        for u in model["utility_nodes"]:
            if "D" in u["parents"]:
                for row in u["table"]:
                    if row["when"]["D"] == acts[-1]:
                        twin = dict(row["when"]); twin["D"] = acts[0]
                        row["value"] = next(r["value"] for r in u["table"] if r["when"] == twin)
    net = Network(model)
    res = DecisionAnalysis(net).sensitivity("T")
    def utilities(p):
        vals = {}
        for a in acts:
            tot = F(0)
            for u in model["utility_nodes"]:
                for t, w in (("hi", p), ("lo", 1 - p)):
                    asg = {"T": t, "D": a}
                    row = next(r for r in u["table"] if all(asg[k] == v for k, v in r["when"].items()))
                    tot += w * F(row["value"])
            vals[a] = tot
        return vals
    regions = res["regions"]
    if regions[0]["from"] != 0 or regions[-1]["to"] != 1:
        return f"sens seed {seed}: regions do not cover [0,1]: {regions}"
    for r1, r2 in zip(regions, regions[1:]):
        if r1["to"] != r2["from"] or r1["optimal_actions"] == r2["optimal_actions"]:
            return f"sens seed {seed}: bad region boundary {r1} {r2}"
    for i in range(1, 2000):
        p = F(i, 2000)
        if any(abs(float(p) - t) < 1e-9 for t in res["thresholds"]):
            continue
        vals = utilities(p)
        best = max(vals.values())
        opt = [a for a in acts if vals[a] == best]
        region = next(r for r in regions if r["from"] <= float(p) <= r["to"])
        if region["optimal_actions"] != opt:
            return f"sens seed {seed}: at p={float(p)} optimal {region['optimal_actions']} vs oracle {opt} (regions {regions})"
    for t in res["thresholds"]:
        lo, hi = utilities(F(t) - F(1, 10**7)), utilities(F(t) + F(1, 10**7))
        if [a for a in acts if lo[a] == max(lo.values())] == [a for a in acts if hi[a] == max(hi.values())]:
            return f"sens seed {seed}: threshold {t} does not change the best action"
    stats["sens"] += 1
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--networks", type=int, default=500, help="Random networks to check (default 500)")
    args = parser.parse_args()
    n = args.networks
    stats = {"prob": 0, "decision": 0, "voi>0": 0, "pervar>0": 0, "sens": 0}
    failures = []
    for seed in range(n):
        try:
            msg = check(seed, stats)
        except Exception:
            msg = f"seed {seed}: crash\n{traceback.format_exc()}"
        if msg:
            failures.append(msg)
        if seed < 400:
            try:
                msg = check_sensitivity(seed, stats)
            except Exception:
                msg = f"sens seed {seed}: crash\n{traceback.format_exc()}"
            if msg:
                failures.append(msg)
    print(f"networks={n} prob_checks={stats['prob']} decision_checks={stats['decision']} voi_positive={stats['voi>0']} per_variable_voi_positive={stats['pervar>0']} sensitivity_checks={stats['sens']} failures={len(failures)}")
    for f in failures[:15]:
        print(f)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
