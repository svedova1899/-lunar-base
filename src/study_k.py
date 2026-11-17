

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layout as L
import sim as S

KS = (0.02, 0.1, 0.5)
res_spread = {K: [] for K in KS}
res_mean = {K: [] for K in KS}
res_exh = {K: [] for K in KS}

for K in KS:
    for seed in range(1, 31):
        lay = L.make_layout(seed)
        r = S.simulate(lay, K=K)
        v = np.array([r["zone_mean"][z] for z in L.ZONES])
        res_spread[K].append(v.max() / v.min())
        res_mean[K].append(v.mean())
    for seed in range(1, 16):
        a = S.simulate(
            L.make_layout(seed, n_stages=2, al_exhaust=False, rotation=0), K=K
        )
        b = S.simulate(
            L.make_layout(seed, n_stages=2, al_exhaust=True, rotation=0), K=K
        )
        va = np.mean([a["zone_mean"][z] for z in L.ZONES])
        vb = np.mean([b["zone_mean"][z] for z in L.ZONES])
        res_exh[K].append(va / vb)
    print(
        "K=%-5s | розкид між зонами (макс/мін), медіана: %.1f (10-90%%: %.1f-%.1f) | "
        "ефект витяжки шлюзу (разів), медіана: %.1f (10-90%%: %.1f-%.1f) | середній пил по 30 планах: %.2f-%.2f мг/м3"
        % (
            K,
            np.median(res_spread[K]),
            *np.percentile(res_spread[K], [10, 90]),
            np.median(res_exh[K]),
            *np.percentile(res_exh[K], [10, 90]),
            np.percentile(res_mean[K], 10),
            np.percentile(res_mean[K], 90),
        )
    )

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
pos = np.arange(len(KS))
axs[0].boxplot([res_spread[K] for K in KS], positions=pos, widths=0.5)
axs[0].set_yscale("log")
axs[0].axhline(1, color="gray", ls=":")
axs[0].set_xticks(pos)
axs[0].set_xticklabels(["K = %g" % K for K in KS])
axs[0].set_ylabel("макс / мін пил між зонами одного плану")
axs[0].set_title("Наскільки зони відрізняються одна від одної\n(30 випадкових планів)")
axs[1].boxplot([res_exh[K] for K in KS], positions=pos, widths=0.5)
axs[1].set_yscale("log")
axs[1].axhline(1, color="gray", ls=":")
axs[1].set_xticks(pos)
axs[1].set_xticklabels(["K = %g" % K for K in KS])
axs[1].set_ylabel("пил без витяжки шлюзу / з витяжкою, разів")
axs[1].set_title("Ефект витяжки в шлюзі (той самий інтер'єр, 15 планів)")
fig.tight_layout()
fig.savefig("out/fig5_dependence_on_K.png", dpi=115)
print("готово")
