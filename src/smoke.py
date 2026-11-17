import time, numpy as np
import layout as L, sim as S

t = time.time()
ok = 0
for seed in range(1, 21):
    try:
        lay = L.make_layout(seed)
        ok += 1
    except Exception as e:
        print("seed", seed, "FAIL", e)
print("layouts ok:", ok, "/20  time %.1fs" % (time.time() - t))

lay = L.make_layout(3, n_stages=2, al_exhaust=False, rotation=0)
print(
    "seed3: stages",
    lay["n_stages"],
    "exh",
    lay["al_exhaust"],
    "supply",
    lay["n_supply"],
    "partitions",
    lay["partitions"],
    "area %.1f m2" % lay["area_m2"],
)
print("path_m", {k: round(v, 1) for k, v in lay["path_m"].items()})
t = time.time()
r = S.simulate(lay)
print(
    "sim time %.1fs" % (time.time() - t),
    "Q=%.3f m3/s V=%.1f m3" % (r["Q_m3s"], r["V_m3"]),
)
print("zone mean 8h (mg/m3):", {k: round(v, 3) for k, v in r["zone_mean"].items()})
m = r["mass"]
print(
    "mass:",
    {
        k: (round(v, 4) if k == "residual_rel" else round(v / 1000, 3))
        for k, v in m.items()
    },
)
print(
    "lam (1/s):",
    r["lam"],
    " v_s mm/s:",
    S.stokes_v(r["params"]["sizes_um"], 1.62) * 1e3,
)
print("max |u| m/s: %.3f" % np.nanmax(np.hypot(r["u"], r["v"])))


room = L.make_simple_room()
t = time.time()
w = S.simulate(room, g=0.0, mass_g=0.0, c0=293.0, t_end=3 * 3600, dt=20.0, snaps=())
ts = w["series"]["t"] / 60
mean = w["series"]["mean"]
mx = w["series"]["max"]


def first_below(y, th=0.3):
    i = np.where(y < th)[0]
    return ts[i[0]] if len(i) else None


print("well-mixed test time %.1fs" % (time.time() - t))
print(
    "time mean<0.3: %s min ; max<0.3: %s min ; analytic %.1f min"
    % (first_below(mean), first_below(mx), np.log(293 / 0.3) / (6.2 * 0.9997 / 60))
)
