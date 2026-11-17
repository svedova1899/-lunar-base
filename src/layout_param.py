"""Параметричний декодер: вектор x in [0,1]^26 -> план бази (той самий формат, що в layout.py).
x[0:10]  позиції 5 зон (u,v) | x[10:16] перегородки сон/гігієна (є?, бік, позиція дверей)
x[16]    1 чи 2 подачі | x[17:21] подачі | x[21:23] витяжка | x[23] ступені | x[24] витяжка шлюзу | x[25] поворот
Повертає (lay, порушення): lay=None, якщо план неможливий."""

import numpy as np
from scipy import ndimage
import layout as L
from layout import N, CX, CY, R, S, DOOR_AL, DOOR_PART, ZONES

SIZES = {
    "сон": (10, 9),
    "їдальня": (10, 9),
    "спорт": (11, 9),
    "гігієна": (8, 8),
    "робота": (10, 9),
}
NX = 26


def decode(x):
    x = np.clip(np.asarray(x, float), 0, 1 - 1e-9)
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - CX, yy - CY)
    fluid = rr <= R - 1
    zone = np.zeros((N, N), np.int8)
    door_in = (CY, CX + R - 2)
    rects = []
    for zi, name in enumerate(ZONES):
        w, h = SIZES[name]
        x0 = int(CX - R + x[2 * zi] * (2 * R - w))
        y0 = int(CY - R + x[2 * zi + 1] * (2 * R - h))
        if not L._rect_in_circle(x0, y0, w, h):
            return None
        ddx = max(x0 - door_in[1], 0, door_in[1] - (x0 + w - 1))
        ddy = max(y0 - door_in[0], 0, door_in[0] - (y0 + h - 1))
        if np.hypot(ddx, ddy) < 5:
            return None
        for r2 in rects:
            gx, gy = L._gap((x0, y0, w, h), r2)
            if not (gx >= 3 or gy >= 3):
                return None
        rects.append((x0, y0, w, h))
    partitions = []
    for zi, (name, (x0, y0, w, h)) in enumerate(zip(ZONES, rects), 1):
        zone[y0 : y0 + h, x0 : x0 + w] = zi
        if name in L.PARTITIONED:
            j = L.PARTITIONED.index(name)
            on, sd, pos = x[10 + 3 * j], x[11 + 3 * j], x[12 + 3 * j]
            if on < 0.5:
                continue
            ring = np.zeros((N, N), bool)
            ring[y0 : y0 + h, x0 : x0 + w] = True
            ring[y0 + 1 : y0 + h - 1, x0 + 1 : x0 + w - 1] = False
            side = int(sd * 4)
            if side in (0, 1):
                xs = x0 + 1 + int(pos * (w - DOOR_PART - 1))
                yd = y0 if side == 0 else y0 + h - 1
                door = [(yd, xx_) for xx_ in range(xs, xs + DOOR_PART)]
            else:
                ys = y0 + 1 + int(pos * (h - DOOR_PART - 1))
                xd = x0 if side == 2 else x0 + w - 1
                door = [(yy_, xd) for yy_ in range(ys, ys + DOOR_PART)]
            fluid[ring] = False
            zone[ring] = 0
            for yy_, xx_ in door:
                fluid[yy_, xx_] = True
                zone[yy_, xx_] = zi
            partitions.append(name)

    def cell(u, v):
        return (
            int(CY - R + 3 + v * (2 * R - 6)),
            int(CX - R + 3 + u * (2 * R - 6)),
        )  

    ns = 1 + int(x[16] >= 0.5)
    sup = [cell(x[17], x[18])] + ([cell(x[19], x[20])] if ns == 2 else [])
    ret = cell(x[21], x[22])
    pts = sup + [ret]
    for y, xx_ in pts:
        if not fluid[y, xx_] or rr[y, xx_] > R - 3:
            return None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < 8:
                return None
    k = 1 + int(x[23] * 3)
    exh = x[24] >= 0.5
    rot = int(x[25] * 4)
    airlock = np.zeros((N, N), np.int8)
    c0 = CX + R + 1
    stages = []
    for j in range(k):
        c1 = c0 + S - 1
        stages.append((c0, c1))
        fluid[CY - S // 2 : CY + S // 2 + 1, c0 : c1 + 1] = True
        airlock[CY - S // 2 : CY + S // 2 + 1, c0 : c1 + 1] = j + 1
        if j < k - 1:
            fluid[CY - DOOR_AL // 2 : CY + DOOR_AL // 2 + 1, c1 + 1] = True
            airlock[CY - DOOR_AL // 2 : CY + DOOR_AL // 2 + 1, c1 + 1] = j + 1
        c0 = c1 + 2
    fluid[CY - DOOR_AL // 2 : CY + DOOR_AL // 2 + 1, CX + R - 1 : CX + R + 1] = True
    z = np.zeros((N, N), bool)
    supply = z.copy()
    ret_m = z.copy()
    alx = z.copy()
    for y, xx_ in sup:
        supply[y, xx_] = True
    ret_m[ret] = True
    if exh:
        alx[CY, (stages[-1][0] + stages[-1][1]) // 2] = True
    source = airlock == k
    door_mask = z.copy()
    door_mask[door_in] = True
    labels, _ = ndimage.label(fluid)
    comp = labels[door_in]
    if comp == 0 or not (labels[zone > 0] == comp).all():
        return None
    arrs = dict(
        fluid=fluid,
        zone=zone,
        airlock=airlock,
        supply=supply,
        ret=ret_m,
        alexh=alx,
        source=source,
        door=door_mask,
    )
    if rot:
        arrs = {key: np.rot90(v, rot).copy() for key, v in arrs.items()}
    door_cell = tuple(int(v) for v in np.argwhere(arrs["door"])[0])
    dist = L._bfs(arrs["fluid"], door_cell)
    path_m = {}
    for zi, name in enumerate(ZONES, 1):
        m = (arrs["zone"] == zi) & (dist >= 0)
        path_m[name] = float(dist[m].mean() * L.DX) if m.any() else float("nan")
    lay = {key: v for key, v in arrs.items() if key != "door"}
    lay.update(
        n_stages=k,
        al_exhaust=bool(exh),
        n_supply=ns,
        partitions=partitions,
        rotation=rot,
        door_cell=door_cell,
        path_m=path_m,
        area_m2=float(lay["fluid"].sum() * L.DX**2),
        volume_m3=float(lay["fluid"].sum() * L.DX**2 * L.H),
        seed=-1,
    )
    return lay


def sample_feasible(rng, tries=4000):
    """Випадковий допустимий вектор: зони розставляються послідовно (як у layout.py), решта генів випадкові."""
    door_in = (CY, CX + R - 2)
    for _ in range(tries):
        x = rng.random(NX)
        rects = []
        good = True
        for zi, name in enumerate(ZONES):
            w, h = SIZES[name]
            placed = False
            for _ in range(200):
                u, v = rng.random(2)
                x0 = int(CX - R + u * (2 * R - w))
                y0 = int(CY - R + v * (2 * R - h))
                if not L._rect_in_circle(x0, y0, w, h):
                    continue
                ddx = max(x0 - door_in[1], 0, door_in[1] - (x0 + w - 1))
                ddy = max(y0 - door_in[0], 0, door_in[0] - (y0 + h - 1))
                if np.hypot(ddx, ddy) < 5:
                    continue
                if any(
                    not (g[0] >= 3 or g[1] >= 3)
                    for g in (L._gap((x0, y0, w, h), r2) for r2 in rects)
                ):
                    continue
                rects.append((x0, y0, w, h))
                x[2 * zi], x[2 * zi + 1] = u, v
                placed = True
                break
            if not placed:
                good = False
                break
        if good and decode(x) is not None:
            return x
    raise RuntimeError("не вдалося")
