
"""Датасет: випадковий план + випадковий K (логарифмічно 0.02-0.5) -> карта log10 пилу."""
import sys, numpy as np, multiprocessing as mp
import layout as L, sim as S

def one(seed):
    rng = np.random.default_rng(seed + 777)
    K = float(np.exp(rng.uniform(np.log(0.02), np.log(0.5))))
    lay = L.make_layout(seed)
    r = S.simulate(lay, K=K)
    x = L.channels(lay)
    kch = np.full((1, L.N, L.N), (np.log10(K) + 1.7) / 1.4, np.float32)   # 0..1
    x = np.concatenate([x, kch])
    c = r["c_avg"]
    y = np.where(lay["fluid"], np.log10(np.nan_to_num(c, nan=0.0) + 1e-4), 0).astype(np.float32)
    zm = np.array([r["zone_mean"][z] for z in L.ZONES], np.float32)
    return x, y, lay["fluid"], K, zm, lay["path_m"], seed

if __name__ == "__main__":
    n = int(sys.argv[1])
    with mp.Pool(2) as p:
        res = p.map(one, range(1, n + 1), chunksize=20)
    X = np.stack([r[0] for r in res]); Y = np.stack([r[1] for r in res])
    F = np.stack([r[2] for r in res]); K = np.array([r[3] for r in res])
    ZM = np.stack([r[4] for r in res]); seeds = np.array([r[6] for r in res])
    np.savez_compressed("out/dataset.npz", X=X, Y=Y, F=F, K=K, ZM=ZM, seeds=seeds)
    print("готово", X.shape, Y.shape, K.min(), K.max())
