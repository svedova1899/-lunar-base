"""NSGA-II над реальними змінними плану; цільові функції рахує навчена U-Net."""

import json, sys, time, pickle, numpy as np, torch
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import layout as L, layout_param as P
from train_unet import UNet

torch.set_num_threads(2)

KS = (0.03, 0.1, 0.4)
meta = json.load(open("out/metrics.json"))
MU, SD = meta["MU"], meta["SD"]
net = UNet()
net.load_state_dict(torch.load("out/unet.pt"))
net.eval()


def zone_dust_surrogate(lays, Ks):
    """-> (len(lays), len(Ks), 5) середній пил по зонах (мг/м3) за прогнозом U-Net."""
    xs = []
    for lay in lays:
        base = L.channels(lay)
        for K in Ks:
            kch = np.full((1, L.N, L.N), (np.log10(K) + 1.7) / 1.4, np.float32)
            xs.append(np.concatenate([base, kch]))
    X = torch.tensor(np.stack(xs))
    with torch.no_grad():
        out = (
            torch.cat([net(X[i : i + 64]) for i in range(0, len(X), 64)]).numpy() * SD
            + MU
        )
    res = np.zeros((len(lays), len(Ks), 5))
    for i, lay in enumerate(lays):
        for j in range(len(Ks)):
            m = out[i * len(Ks) + j]
            for z in range(5):
                res[i, j, z] = (10 ** m[lay["zone"] == z + 1]).mean()
    return res


def objectives_from_dust(lay, dust):

    sleep_hyg = dust[:, [0, 3]].mean(1)
    other = dust[:, [1, 2, 4]].mean(1)
    f1 = sleep_hyg.mean()
    f2 = other.mean()
    f3 = float(np.nanmean(list(lay["path_m"].values())))
    f4 = lay["n_stages"] + 0.5 * lay["al_exhaust"]
    return [f1, f2, f3, f4]


class Prob(Problem):
    def __init__(s):
        super().__init__(n_var=P.NX, n_obj=4, xl=0.0, xu=1.0)
        s.pool = []

    def _evaluate(s, X, out, *a, **k):
        lays = [P.decode(x) for x in X]
        assert all(l is not None for l in lays), "репарація мала дати допустимі плани"
        dust = zone_dust_surrogate(lays, KS)
        out["F"] = np.array([objectives_from_dust(l, d) for l, d in zip(lays, dust)])
        s.pool = [x.copy() for x in X]


class Samp(Sampling):
    def _do(s, problem, n_samples, **k):
        rng = np.random.default_rng(0)
        return np.stack([P.sample_feasible(rng) for _ in range(n_samples)])


class Rep(Repair):
    """Недопустимий нащадок зсувається до допустимого плану з поточної популяції."""

    def __init__(s, prob):
        super().__init__()
        s.prob = prob
        s.rng = np.random.default_rng(5)

    def _do(s, problem, X, **k):
        X = X.copy()
        ok = [x for x in X if P.decode(x) is not None]
        pool = s.prob.pool or ok
        for i, x in enumerate(X):
            if P.decode(x) is not None:
                continue
            ref = pool[int(s.rng.integers(len(pool)))]
            for t in (0.75, 0.5, 0.3, 0.15, 0.07, 0.0):
                y = ref + t * (x - ref)
                if P.decode(y) is not None:
                    X[i] = y
                    break
        return X


if __name__ == "__main__":
    pop, gens = int(sys.argv[1]), int(sys.argv[2])
    prob = Prob()
    alg = NSGA2(
        pop_size=pop,
        sampling=Samp(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        repair=Rep(prob),
        eliminate_duplicates=False,
    )
    t0 = time.time()
    res = minimize(
        prob, alg, ("n_gen", gens), seed=1, verbose=False, save_history=False
    )
    print("NSGA-II: %.0f с, фронт Парето: %d планів" % (time.time() - t0, len(res.F)))
    pickle.dump(dict(X=res.X, F=res.F), open("out/pareto_surrogate.pkl", "wb"))
    print(
        "f1 мін/макс: %.3f %.3f | f2: %.3f %.3f"
        % (res.F[:, 0].min(), res.F[:, 0].max(), res.F[:, 1].min(), res.F[:, 1].max())
    )
