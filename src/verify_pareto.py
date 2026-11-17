
"""Перевірка фронту Парето справжнім симулятором + порівняння з випадковими планами."""

import pickle, numpy as np, multiprocessing as mp
import layout as L, layout_param as P, sim as S
from optimize_nsga2 import KS, objectives_from_dust, zone_dust_surrogate


def true_obj(x):
    lay = P.decode(x)
    dust = np.array(
        [[S.simulate(lay, K=K)["zone_mean"][z] for z in L.ZONES] for K in KS]
    )
    return objectives_from_dust(lay, dust)


if __name__ == "__main__":
    d = pickle.load(open("out/pareto_surrogate.pkl", "rb"))
    Xp, Fs = d["X"], d["F"]
    rng = np.random.default_rng(42)
    Xr = np.stack([P.sample_feasible(rng) for _ in range(150)])
    with mp.Pool(2) as pool:
        Ft = np.array(pool.map(true_obj, list(Xp), chunksize=5))
        Frt = np.array(pool.map(true_obj, list(Xr), chunksize=5))
    Frs = np.array(
        [
            objectives_from_dust(l, dd)
            for l, dd in zip(
                [P.decode(x) for x in Xr],
                zone_dust_surrogate([P.decode(x) for x in Xr], KS),
            )
        ]
    )
    pickle.dump(
        dict(Xp=Xp, Fs=Fs, Ft=Ft, Xr=Xr, Frt=Frt, Frs=Frs), open("out/verify.pkl", "wb")
    )
    from scipy.stats import spearmanr

    for i, nm in enumerate(
        ["f1 (сон+гігієна, середнє по K)", "f2 (інші зони, середнє по K)"]
    ):
        r = spearmanr(Fs[:, i], Ft[:, i]).correlation
        fac = np.exp(np.abs(np.log(Fs[:, i] / Ft[:, i])))
        print(
            "%s: Spearman(U-Net, симулятор) на фронті = %.2f; типова похибка ×%.2f; U-Net занижує пил у %d з %d"
            % (nm, r, np.median(fac), (Fs[:, i] < Ft[:, i]).sum(), len(Fs))
        )
        rr = spearmanr(Frs[:, i], Frt[:, i]).correlation
        print("   на випадкових планах: Spearman = %.2f" % rr)
    print(
        "справжній симулятор: випадкові плани f1 медіана %.2f (10-90%%: %.2f-%.2f)"
        % (np.median(Frt[:, 0]), *np.percentile(Frt[:, 0], [10, 90]))
    )
    print(
        "                     фронт Парето    f1 медіана %.2f (10-90%%: %.2f-%.2f), мінімум %.2f"
        % (np.median(Ft[:, 0]), *np.percentile(Ft[:, 0], [10, 90]), Ft[:, 0].min())
    )
    print(
        "випадкові f2 медіана %.2f | фронт f2 медіана %.2f, мінімум %.2f"
        % (np.median(Frt[:, 1]), np.median(Ft[:, 1]), Ft[:, 1].min())
    )
    print("кращий випадковий f1: %.2f" % Frt[:, 0].min())
