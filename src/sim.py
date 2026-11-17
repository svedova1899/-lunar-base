# -*- coding: utf-8 -*-
"""Симулятор перенесення місячного пилу всередині бази (2D, усереднено по висоті).

Модель (для кожного розміру частинок окремо):

    dc/dt + div(u c) = div(K grad c) - (v_s / H) c + джерело
    (рівняння перенесення-розсіювання з осіданням)

  * u      — потік повітря: потенціальна течія від вентиляції (джерела — подача,
             стоки — витяжки). Вихорів і турбулентності немає, їх замінює
             ефективна дифузія K (допущення);
  * v_s    — швидкість осідання за Стоксом при g = 1,62 м/с2;
  * витяжка проходить через HEPA (ефективність eta); частка (1-eta) повертається
    на подачу (рециркуляція);
  * джерело — пил, що став аерозолем у «брудній» камері шлюзу за перші
    release_s секунд після виходу екіпажу.

Розв'язується неявною схемою (безумовно стійка, додатні концентрації).
Ведеться облік маси: випущено = у повітрі + осіло + затримано фільтрами.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve
from scipy.sparse.csgraph import connected_components

from layout import DX, H, ZONES

RHO_P = 3100.0    # кг/м3, щільність частинки реголіту
MU_AIR = 1.8e-5   # Па*с, в'язкість повітря

DEFAULTS = dict(
    g=1.62,                    # м/с2
    K=0.02,                    # м2/с, ефективна дифузія (допущення)
    ach=6.2,                   # обмінів повітря за годину (приклад МКС у [22])
    eta=0.9997,                # ефективність HEPA
    f_airlock=0.15,            # частка потоку, що відсмоктується зі шлюзу (якщо є витяжка)
    sizes_um=(1.0, 3.0, 10.0),
    fractions=(0.3, 0.4, 0.3),  # допущення про розподіл за розмірами
    mass_g=31.8,               # 2 людини * 227 г * 7 % ([22])
    release_s=1200.0,          # викид за перші 20 хв
    t_end=8 * 3600.0,
    dt=30.0,
    snaps=(300.0, 1800.0, 7200.0, 28800.0),
    c0=None,                   # початкова рівномірна концентрація, мг/м3 (для перевірок)
    rec_every=4,
)


def stokes_v(d_um, g):
    """Швидкість осідання сфери за Стоксом, м/с (без поправки Каннінгема)."""
    d = np.asarray(d_um, float) * 1e-6
    return RHO_P * d * d * g / (18.0 * MU_AIR)


def _pairs(fluid):
    idx = -np.ones(fluid.shape, dtype=np.int64)
    n = int(fluid.sum())
    idx[fluid] = np.arange(n)
    mx = fluid[:, :-1] & fluid[:, 1:]
    my = fluid[:-1, :] & fluid[1:, :]
    ax_, bx_ = idx[:, :-1][mx], idx[:, 1:][mx]
    ay_, by_ = idx[:-1, :][my], idx[1:, :][my]
    a = np.concatenate([ax_, ay_])
    b = np.concatenate([bx_, by_])
    return idx, n, a, b, len(ax_)


def _solve_phi(n, a, b, rhs):
    """Потенціал швидкості: sum_j (phi_j - phi_i) = rhs_i (подача - витяжка)."""
    adj = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    L = (adj - sp.diags(deg)).tocsr()
    ncomp, lab = connected_components(adj, directed=False)
    pins = []
    for c in range(ncomp):
        mem = np.flatnonzero(lab == c)
        pins.append(int(mem[0]))
        assert abs(rhs[mem].sum()) < 1e-9 * max(1.0, np.abs(rhs).sum()), \
            "джерела й стоки не збалансовані в одній зв'язній області"
    keep = np.ones(n, bool)
    keep[pins] = False
    phi = np.zeros(n)
    phi[keep] = spsolve(L[keep][:, keep].tocsc(), rhs[keep])
    return phi


def _cell_velocity(fluid, F, nxp):
    ny, nx = fluid.shape
    ux = np.zeros((ny, nx)); vy = np.zeros((ny, nx))
    cu = np.zeros((ny, nx)); cv = np.zeros((ny, nx))
    ia = np.argwhere(fluid[:, :-1] & fluid[:, 1:])
    ja = np.argwhere(fluid[:-1, :] & fluid[1:, :])
    Fx, Fy = F[:nxp], F[nxp:]
    for (y, x), f in zip(ia, Fx):
        ux[y, x] += f; ux[y, x + 1] += f; cu[y, x] += 1; cu[y, x + 1] += 1
    for (y, x), f in zip(ja, Fy):
        vy[y, x] += f; vy[y + 1, x] += f; cv[y, x] += 1; cv[y + 1, x] += 1
    return ux / np.maximum(cu, 1) / DX, vy / np.maximum(cv, 1) / DX   # м/с


def _transport(n, a, b, F, S_out, K, lam):
    A2 = DX * DX
    Fp = np.maximum(F, 0.0)
    Fm = np.maximum(-F, 0.0)
    diag = (np.bincount(a, Fp, n) + np.bincount(b, Fm, n)) / A2
    deg = (np.bincount(a, minlength=n) + np.bincount(b, minlength=n)).astype(float)
    kd = K / A2
    diag = diag + kd * deg + S_out / A2 + lam
    rows = np.concatenate([b, a, a, b])
    cols = np.concatenate([a, b, b, a])
    vals = np.concatenate([-Fp / A2, -Fm / A2, -kd * np.ones(len(a)), -kd * np.ones(len(a))])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr() + sp.diags(diag)
    return A.tocsc()


def simulate(lay, **kw):
    p = {**DEFAULTS, **kw}
    fluid = lay["fluid"]
    idx, n, a, b, nxp = _pairs(fluid)
    A2 = DX * DX
    V = n * A2 * H
    Q = p["ach"] * V / 3600.0          # м3/с
    q = Q / H                           # м2/с на одиницю глибини

    sup = idx[lay["supply"]]
    ret = idx[lay["ret"]]
    alx = idx[lay["alexh"]]
    S_in = np.zeros(n); S_out = np.zeros(n)
    S_in[sup] = q / len(sup)
    if len(alx):
        S_out[alx] = p["f_airlock"] * q / len(alx)
        S_out[ret] = (1.0 - p["f_airlock"]) * q / len(ret)
    else:
        S_out[ret] = q / len(ret)

    phi = _solve_phi(n, a, b, S_in - S_out)
    F = phi[b] - phi[a]                 # потік a -> b, м2/с
    ux, vy = _cell_velocity(fluid, F, nxp)

    sizes = np.asarray(p["sizes_um"], float)
    fr = np.asarray(p["fractions"], float)
    nb = len(sizes)
    lam = stokes_v(sizes, p["g"]) / H   # 1/с
    dt = p["dt"]
    Id = sp.identity(n, format="csc") / dt
    lus = [splu((Id + _transport(n, a, b, F, S_out, p["K"], lam[k])).tocsc()) for k in range(nb)]

    src = idx[lay["source"]]
    rate = p["mass_g"] * 1000.0 / p["release_s"]       # мг/с
    if p["mass_g"] > 0 and len(src) == 0:
        raise ValueError("є маса викиду, але немає клітинок джерела")
    amp = rate * fr / (max(len(src), 1) * A2 * H)      # мг/м3/с для кожного біна

    def grid(vec):
        g = np.full(fluid.shape, np.nan)
        g[fluid] = vec
        return g

    nsteps = int(round(p["t_end"] / dt))
    c = np.zeros((nb, n))
    if p["c0"] is not None:
        c[:] = (fr * p["c0"])[:, None]
    M0 = H * A2 * c.sum()
    cavg = np.zeros(n)
    Sout_sum = S_out.sum()
    snap_steps = {int(round(t / dt)): t for t in p["snaps"]}
    snaps = {}
    zone_idx = {name: idx[lay["zone"] == zi] for zi, name in enumerate(ZONES, 1)}
    ts, smean, smax = [], [], []
    szone = {name: [] for name in ZONES}
    released = recirc = vent_out = settled = 0.0

    for step in range(nsteps):
        t = step * dt
        csup = (1.0 - p["eta"]) * (c @ S_out) / Sout_sum       # (nb,) концентрація на подачі
        rhs = c / dt
        if t < p["release_s"] and p["mass_g"] > 0:
            rhs[:, src] += amp[:, None]
            released += rate * dt
        rhs[:, sup] += (S_in[sup] / A2)[None, :] * csup[:, None]
        recirc += dt * H * q * float(csup.sum())
        for k in range(nb):
            c[k] = lus[k].solve(rhs[k])
        vent_out += dt * H * float((c @ S_out).sum())
        settled += dt * H * A2 * float((lam[:, None] * c).sum())
        tot = c.sum(axis=0)
        cavg += tot * dt
        if (step + 1) in snap_steps:
            snaps[snap_steps[step + 1]] = grid(tot)
        if (step + 1) % p["rec_every"] == 0:
            ts.append((step + 1) * dt)
            smean.append(float(tot.mean()))
            smax.append(float(tot.max()))
            for name, ii in zone_idx.items():
                szone[name].append(float(tot[ii].mean()) if len(ii) else np.nan)

    cavg /= (nsteps * dt)
    M_end = H * A2 * float(c.sum())
    resid = M_end - (M0 + released + recirc - vent_out - settled)
    total_in = max(M0 + released, 1e-12)
    zone_mean = {name: (float(cavg[ii].mean()) if len(ii) else np.nan) for name, ii in zone_idx.items()}
    return dict(
        c_avg=grid(cavg), snaps=snaps, zone_mean=zone_mean,
        series=dict(t=np.array(ts), mean=np.array(smean), max=np.array(smax),
                    zone={k: np.array(v) for k, v in szone.items()}),
        mass=dict(released_mg=released + M0, in_air_mg=M_end, settled_mg=settled,
                  filtered_mg=vent_out - recirc, residual_rel=resid / total_in),
        u=ux, v=vy, Q_m3s=Q, V_m3=V, lam=lam, params=p,
    )
