

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D
import layout as L
import sim as S

OUT = "out"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

ZCOL = {
    "сон": "#4C78A8",
    "їдальня": "#F58518",
    "спорт": "#54A24B",
    "гігієна": "#B279A2",
    "робота": "#E45756",
}
LIMIT = 0.3  # мг/м3, орієнтир [6, 7]


def bbox(lay, m=2):
    ys, xs = np.nonzero(lay["fluid"])
    return (
        max(xs.min() - m, 0),
        min(xs.max() + m, L.N - 1),
        max(ys.min() - m, 0),
        min(ys.max() + m, L.N - 1),
    )


def extent(b):
    x0, x1, y0, y1 = b
    return [
        (x0 - L.CX - 0.5) * L.DX,
        (x1 - L.CX + 0.5) * L.DX,
        (y0 - L.CY - 0.5) * L.DX,
        (y1 - L.CY + 0.5) * L.DX,
    ]


def xy(mask):
    p = np.argwhere(mask)
    return (p[:, 1] - L.CX) * L.DX, (p[:, 0] - L.CY) * L.DX


def markers(ax, lay):
    x, y = xy(lay["supply"])
    ax.plot(x, y, "^", color="#1f77b4", ms=8, mec="k", zorder=5)
    x, y = xy(lay["ret"])
    ax.plot(x, y, "v", color="#d62728", ms=8, mec="k", zorder=5)
    if lay["alexh"].any():
        x, y = xy(lay["alexh"])
        ax.plot(x, y, "D", color="#9467bd", ms=7, mec="k", zorder=5)


MARK_LEGEND = [
    Line2D([], [], marker="^", ls="", color="#1f77b4", mec="k", label="подача повітря"),
    Line2D([], [], marker="v", ls="", color="#d62728", mec="k", label="витяжка + HEPA"),
    Line2D(
        [],
        [],
        marker="D",
        ls="",
        color="#9467bd",
        mec="k",
        label="витяжка шлюзу + HEPA",
    ),
]


def zone_labels(ax, lay, fs=7):
    for zi, name in enumerate(L.ZONES, 1):
        m = (lay["zone"] == zi) & lay["fluid"]
        if m.any():
            x, y = xy(m)
            ax.text(
                x.mean(),
                y.mean(),
                name,
                ha="center",
                va="center",
                fontsize=fs,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                zorder=6,
            )


def draw_layout(ax, lay, labels=True):
    b = bbox(lay)
    x0, x1, y0, y1 = b
    rgb = np.ones((L.N, L.N, 3)) * 0.25
    base = np.array([0.95, 0.94, 0.91])
    rgb[lay["fluid"]] = base
    for zi, name in enumerate(L.ZONES, 1):
        m = (lay["zone"] == zi) & lay["fluid"]
        rgb[m] = 0.55 * base + 0.45 * np.array(to_rgb(ZCOL[name]))
    rgb[(lay["airlock"] > 0) & lay["fluid"]] = [0.82, 0.86, 0.92]
    ax.imshow(
        rgb[y0 : y1 + 1, x0 : x1 + 1],
        origin="lower",
        extent=extent(b),
        interpolation="nearest",
    )
    markers(ax, lay)
    if labels:
        zone_labels(ax, lay)
    ax.set_xlabel("м")
    ax.set_ylabel("м")


def draw_field(ax, lay, field, norm, cmap="viridis", contour=True):
    b = bbox(lay)
    x0, x1, y0, y1 = b
    f = np.ma.masked_invalid(field)[y0 : y1 + 1, x0 : x1 + 1]
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#3b3b3b")
    im = ax.imshow(
        f, origin="lower", extent=extent(b), cmap=cm, norm=norm, interpolation="nearest"
    )
    ff = np.ma.masked_invalid(field[y0 : y1 + 1, x0 : x1 + 1])
    if contour and ff.max() > LIMIT > ff.min():
        X = (np.arange(x0, x1 + 1) - L.CX) * L.DX
        Y = (np.arange(y0, y1 + 1) - L.CY) * L.DX
        ax.contour(
            X, Y, ff, levels=[LIMIT], colors="white", linewidths=1.0, linestyles="--"
        )
    markers(ax, lay)
    ax.set_xlabel("м")
    ax.set_ylabel("м")
    return im


print("Рис.0: приклади планів")
fig, axs = plt.subplots(2, 3, figsize=(13, 8.5))
for ax, seed in zip(axs.ravel(), range(1, 7)):
    lay = L.make_layout(seed)
    draw_layout(ax, lay)
    ax.set_title(
        "seed %d: ступенів шлюзу %d, витяжка в шлюзі: %s\nперегородки: %s"
        % (
            seed,
            lay["n_stages"],
            "є" if lay["al_exhaust"] else "немає",
            ", ".join(lay["partitions"]) or "немає",
        ),
        fontsize=8.5,
    )
axs[0, 0].legend(handles=MARK_LEGEND, loc="lower left", fontsize=7)
fig.suptitle("Приклади випадкових планів (темне — стіни, блакитне — камери шлюзу)")
fig.tight_layout()
fig.savefig(OUT + "/fig0_generator_examples.png", dpi=120)
plt.close(fig)


SEED = 3
configs = [
    ("1 ступінь, без витяжки в шлюзі", dict(n_stages=1, al_exhaust=False)),
    ("3 ступені, без витяжки в шлюзі", dict(n_stages=3, al_exhaust=False)),
    ("3 ступені + витяжка в шлюзі", dict(n_stages=3, al_exhaust=True)),
]
runs = []
for name, kw in configs:
    lay = L.make_layout(SEED, rotation=0, **kw)
    r = S.simulate(lay)
    runs.append((name, lay, r))
    print(
        "%-34s" % name,
        {k: round(v, 2) for k, v in r["zone_mean"].items()},
        "| маса: осіло %.1f%%, фільтри %.1f%%, у повітрі %.2f%%, нев'язка %.1e"
        % (
            100 * r["mass"]["settled_mg"] / r["mass"]["released_mg"],
            100 * r["mass"]["filtered_mg"] / r["mass"]["released_mg"],
            100 * r["mass"]["in_air_mg"] / r["mass"]["released_mg"],
            r["mass"]["residual_rel"],
        ),
    )


print("Рис.1: план і потік повітря")
name, lay, r = runs[2]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
draw_layout(axs[0], lay)
axs[0].set_title("План: " + name)
axs[0].legend(handles=MARK_LEGEND, loc="lower left", fontsize=7)
b = bbox(lay)
x0, x1, y0, y1 = b
speed = np.hypot(r["u"], r["v"])
speed[~lay["fluid"]] = np.nan
cm = plt.get_cmap("cividis").copy()
cm.set_bad("#3b3b3b")
im = axs[1].imshow(
    np.ma.masked_invalid(speed)[y0 : y1 + 1, x0 : x1 + 1],
    origin="lower",
    extent=extent(b),
    cmap=cm,
    norm=LogNorm(1e-4, 1e-1),
    interpolation="nearest",
)
X = (np.arange(x0, x1 + 1) - L.CX) * L.DX
Y = (np.arange(y0, y1 + 1) - L.CY) * L.DX
uu = np.where(lay["fluid"], r["u"], 0.0)[y0 : y1 + 1, x0 : x1 + 1]
vv = np.where(lay["fluid"], r["v"], 0.0)[y0 : y1 + 1, x0 : x1 + 1]
slow = np.hypot(uu, vv) < 3e-4  
uu = np.where(slow, 0.0, uu)
vv = np.where(slow, 0.0, vv)
axs[1].streamplot(
    X, Y, uu, vv, density=2.0, color="white", linewidth=0.6, arrowsize=0.8
)
markers(axs[1], lay)
axs[1].set_title(
    "Потік повітря: Q = %.2f м³/с, %.1f обмінів/год\n(лінії не малюються, де швидкість < 0,3 мм/с)"
    % (r["Q_m3s"], r["params"]["ach"])
)
axs[1].set_xlabel("м")
axs[1].set_ylabel("м")
fig.colorbar(im, ax=axs[1], fraction=0.046, label="швидкість, м/с")
fig.tight_layout()
fig.savefig(OUT + "/fig1_layout_and_flow.png", dpi=120)
plt.close(fig)


print("Рис.2: пил у часі")
name, lay, r = runs[1]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.8))
norm = LogNorm(1e-3, 1e3)
for ax, (t, f) in zip(axs, sorted(r["snaps"].items())):
    im = draw_field(ax, lay, f, norm)
    ax.set_title("%s" % ("%d хв" % (t / 60) if t < 3600 * 3 else "%d год" % (t / 3600)))
fig.suptitle(
    "Пил у часі після виходу екіпажу (%s); біла пунктирна лінія — 0,3 мг/м³" % name
)
fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, label="концентрація, мг/м³ (лог.)")
fig.savefig(OUT + "/fig2_dust_in_time.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Рис. 3
print("Рис.3: ефект шлюзу")
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
norm = LogNorm(1e-2, 1e3)
for j, (name, lay, r) in enumerate(runs):
    ax = fig.add_subplot(gs[0, j])
    im = draw_field(ax, lay, r["c_avg"], norm)
    zone_labels(ax, lay, fs=6.5)
    ax.set_title(name)
fig.colorbar(
    im, ax=fig.axes[:3], fraction=0.02, pad=0.01, label="середня за 8 год, мг/м³ (лог.)"
)
ax = fig.add_subplot(gs[1, 0:2])
w = 0.26
for j, (name, lay, r) in enumerate(runs):
    vals = [r["zone_mean"][z] for z in L.ZONES]
    ax.bar(np.arange(len(L.ZONES)) + (j - 1) * w, vals, w, label=name)
ax.set_yscale("log")
ax.axhline(LIMIT, color="k", ls="--", lw=1)
ax.text(4.45, LIMIT * 1.15, "0,3 мг/м³ (орієнтир)", ha="right", fontsize=8)
ax.set_xticks(np.arange(len(L.ZONES)))
ax.set_xticklabels(L.ZONES)
ax.set_ylabel("середня за 8 год, мг/м³")
ax.legend(fontsize=8)
ax.set_title("Пил по зонах (однаковий інтер'єр, різні шлюзи)")
ax = fig.add_subplot(gs[1, 2])
bot = np.zeros(3)
for key, lab, col in (
    ("settled_mg", "осіло на підлогу", "#8c6d31"),
    ("filtered_mg", "затримано HEPA", "#1f77b4"),
    ("in_air_mg", "лишилось у повітрі", "#d62728"),
):
    v = np.array([100 * r["mass"][key] / r["mass"]["released_mg"] for _, _, r in runs])
    ax.bar(range(3), v, bottom=bot, label=lab, color=col)
    bot += v
ax.set_xticks(range(3))
ax.set_xticklabels(["1 ступ.", "3 ступ.", "3 ступ.+\nвитяжка"])
ax.set_ylim(0, 135)
ax.set_ylabel("% від викинутої маси")
ax.set_title("Баланс маси через 8 год")
ax.legend(fontsize=8, loc="upper center")
fig.savefig(OUT + "/fig3_airlock_effect.png", dpi=110, bbox_inches="tight")
plt.close(fig)

print("Рис.4: перевірки")
room = L.make_simple_room()
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axs[0]
t_an = np.linspace(0, 120, 400)
ax.plot(
    t_an,
    293 * np.exp(-6.2 * 0.9997 / 60 * t_an),
    "k-",
    lw=2,
    label="ідеальне перемішування: 293·exp(−η·Q·t/V)",
)
print("Перевірка перемішування (час до середньої <0,3 мг/м³):")
for K in (0.02, 0.2, 2.0):
    w_ = S.simulate(
        room, g=0.0, mass_g=0.0, c0=293.0, t_end=2 * 3600, dt=20.0, snaps=(), K=K
    )
    ts = w_["series"]["t"] / 60
    i = np.where(w_["series"]["mean"] < LIMIT)[0]
    t03 = ts[i[0]] if len(i) else float("nan")
    print("   K = %-5s -> %.0f хв" % (K, t03))
    ax.plot(
        ts,
        w_["series"]["mean"],
        label="симулятор, K=%g м²/с (0,3 за %.0f хв)" % (K, t03),
    )
ax.axhline(LIMIT, color="gray", ls=":")
ax.axvline(67, color="r", ls="--", lw=1)
ax.text(68, 1e3, "67 хв за [22]", color="r", fontsize=8)
ax.set_yscale("log")
ax.set_ylim(1e-2, 5e2)
ax.set_xlabel("хв")
ax.set_ylabel("середня концентрація, мг/м³")
ax.set_title("Перевірка: кімната 6,2 обм./год, початково 293 мг/м³")
ax.legend(fontsize=7)

ax = axs[1]
d = np.logspace(0, 1.4, 100)
for g, c in ((1.62, "#1f77b4"), (9.81, "#d62728")):
    ax.plot(d, S.H / S.stokes_v(d, g) / 60, color=c, label="g = %.2f м/с²" % g)
for dd in (1, 3, 10):
    ax.plot(dd, S.H / S.stokes_v(dd, 1.62) / 60, "ko")
ax.set_xscale("log")
ax.set_yscale("log")
ax.axhspan(30, 60, color="orange", alpha=0.25)
ax.text(1.05, 40, "30–60 хв: оцінка [22] для 10–20 мкм", fontsize=8)
ax.set_xlabel("діаметр частинки, мкм")
ax.set_ylabel("час падіння з %.1f м, хв" % S.H)
ax.set_title("Осідання за Стоксом")
ax.legend(fontsize=8)
print(
    "Осідання з 2,5 м на Місяці: 1 мкм %.0f год, 3 мкм %.1f год, 10 мкм %.0f хв"
    % (
        S.H / S.stokes_v(1, 1.62) / 3600,
        S.H / S.stokes_v(3, 1.62) / 3600,
        S.H / S.stokes_v(10, 1.62) / 60,
    )
)

ax = axs[2]
lay = runs[1][1]
for K in (0.02, 0.1, 0.5):
    r = S.simulate(lay, K=K)
    ax.plot(
        L.ZONES, [r["zone_mean"][z] for z in L.ZONES], "o-", label="K = %g м²/с" % K
    )
    print(
        "Чутливість до K=%g:" % K, {k: round(v, 2) for k, v in r["zone_mean"].items()}
    )
ax.set_yscale("log")
ax.axhline(LIMIT, color="k", ls="--", lw=1)
ax.set_ylabel("середня за 8 год, мг/м³")
ax.set_title("Чутливість зон до дифузії K\n(план: 3 ступені, без витяжки)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT + "/fig4_checks.png", dpi=110)
plt.close(fig)
print("готово")
