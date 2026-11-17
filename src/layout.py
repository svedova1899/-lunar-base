"""Генератор планів друкованої місячної бази (2D, вигляд зверху).

Об'єкт: герметичний (надувний) об'єм під друкованим реголітовим куполом
за концепцією ESA / Foster+Partners + шлюз із 1-3 ступенів, що примикає до
купола. Усе на сітці N x N з кроком DX.

Припущення (спрощення авторки):
  * план у перерізі на висоті людини, усереднення по висоті H;
  * внутрішнє планування концепції ESA публічно не деталізоване, тому
    зони, перегородки й вентиляція задаються випадково й параметрично.
"""

import numpy as np
from collections import deque
from scipy import ndimage

N = 96
DX = 0.2 
H = 2.5 
CX = CY = N // 2
R = 22 
S = 7  
DOOR_AL = 5  
DOOR_PART = 4  

ZONES = ["сон", "їдальня", "спорт", "гігієна", "робота"]
ZONE_DIMS = {  
    "сон": (9, 12, 8, 11),
    "їдальня": (9, 12, 8, 10),
    "спорт": (9, 13, 8, 10),
    "гігієна": (7, 9, 7, 9),
    "робота": (9, 12, 8, 10),
}
PARTITIONED = ("сон", "гігієна")


def _rect_in_circle(x0, y0, w, h, margin=3):
    for x, y in (
        (x0, y0),
        (x0 + w - 1, y0),
        (x0, y0 + h - 1),
        (x0 + w - 1, y0 + h - 1),
    ):
        if np.hypot(x - CX, y - CY) > R - margin:
            return False
    return True


def _gap(r1, r2):
    x0, y0, w, h = r1
    X0, Y0, W, Hh = r2
    gx = max(X0 - (x0 + w), x0 - (X0 + W))
    gy = max(Y0 - (y0 + h), y0 - (Y0 + Hh))
    return gx, gy


def _bfs(fluid, start):
    dist = np.full(fluid.shape, -1, np.int32)
    dist[start] = 0
    dq = deque([start])
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < N and 0 <= nx < N and fluid[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                dq.append((ny, nx))
    return dist


def _try_build(rng, k, exh, rot, p_part):
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.hypot(xx - CX, yy - CY)
    fluid = rr <= R - 1
    zone = np.zeros((N, N), np.int8)
    door_in = (CY, CX + R - 2)


    rects = []
    for zi, name in enumerate(ZONES, 1):
        wmin, wmax, hmin, hmax = ZONE_DIMS[name]
        placed = None
        for _ in range(300):
            w = int(rng.integers(wmin, wmax + 1))
            h = int(rng.integers(hmin, hmax + 1))
            x0 = int(rng.integers(CX - R, CX + R))
            y0 = int(rng.integers(CY - R, CY + R))
            if not _rect_in_circle(x0, y0, w, h):
                continue
            ddx = max(x0 - door_in[1], 0, door_in[1] - (x0 + w - 1))
            ddy = max(y0 - door_in[0], 0, door_in[0] - (y0 + h - 1))
            if np.hypot(ddx, ddy) < 5:
                continue
            ok = True
            for r2 in rects:
                gx, gy = _gap((x0, y0, w, h), r2)
                if not (gx >= 3 or gy >= 3):
                    ok = False
                    break
            if ok:
                placed = (x0, y0, w, h)
                break
        if placed is None:
            return None
        rects.append(placed)

    partitions = []
    for zi, (name, (x0, y0, w, h)) in enumerate(zip(ZONES, rects), 1):
        zone[y0 : y0 + h, x0 : x0 + w] = zi
        if name in PARTITIONED and rng.random() < p_part:
            ring = np.zeros((N, N), bool)
            ring[y0 : y0 + h, x0 : x0 + w] = True
            ring[y0 + 1 : y0 + h - 1, x0 + 1 : x0 + w - 1] = False
            side = int(rng.integers(0, 4))
            if side in (0, 1):
                xs = int(rng.integers(x0 + 1, x0 + w - DOOR_PART))
                yd = y0 if side == 0 else y0 + h - 1
                door = [(yd, x) for x in range(xs, xs + DOOR_PART)]
            else:
                ys = int(rng.integers(y0 + 1, y0 + h - DOOR_PART))
                xd = x0 if side == 2 else x0 + w - 1
                door = [(y, xd) for y in range(ys, ys + DOOR_PART)]
            fluid[ring] = False
            zone[ring] = 0
            for y, x in door:
                fluid[y, x] = True
                zone[y, x] = zi
            partitions.append(name)

    cand = np.argwhere(fluid & (rr <= R - 3))

    def pick(n, avoid):
        out = []
        for _ in range(300):
            y, x = cand[int(rng.integers(len(cand)))]
            if all(np.hypot(y - a, x - b) >= 8 for a, b in list(avoid) + out):
                out.append((int(y), int(x)))
                if len(out) == n:
                    return out
        return None

    sup = pick(int(rng.integers(1, 3)), [])
    if sup is None:
        return None
    ret = pick(1, sup)
    if ret is None:
        return None

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

    supply = np.zeros((N, N), bool)
    for y, x in sup:
        supply[y, x] = True
    ret_m = np.zeros((N, N), bool)
    ret_m[ret[0]] = True
    alx = np.zeros((N, N), bool)
    if exh:
        alx[CY, (stages[-1][0] + stages[-1][1]) // 2] = True
    source = airlock == k
    door_mask = np.zeros((N, N), bool)
    door_mask[door_in] = True

    labels, _ = ndimage.label(fluid)
    comp = labels[door_in]
    if comp == 0 or not (labels[zone > 0] == comp).all():
        return None
    for m in (supply, ret_m, alx, source):
        if m.any() and not (labels[m] == comp).all():
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
    dist = _bfs(arrs["fluid"], door_cell)
    path_m = {}
    for zi, name in enumerate(ZONES, 1):
        m = (arrs["zone"] == zi) & (dist >= 0)
        path_m[name] = float(dist[m].mean() * DX) if m.any() else float("nan")

    lay = {key: v for key, v in arrs.items() if key != "door"}
    lay.update(
        n_stages=k,
        al_exhaust=bool(exh),
        n_supply=len(sup),
        partitions=partitions,
        rotation=rot,
        door_cell=door_cell,
        path_m=path_m,
        area_m2=float(lay["fluid"].sum() * DX * DX),
        volume_m3=float(lay["fluid"].sum() * DX * DX * H),
    )
    return lay


def make_layout(
    seed, n_stages=None, al_exhaust=None, rotation=None, partition_prob=0.5
):
    """Той самий seed дає той самий інтер'єр незалежно від параметрів шлюзу."""
    rng_al = np.random.default_rng(seed + 100003)
    k = int(rng_al.integers(1, 4)) if n_stages is None else int(n_stages)
    exh = bool(rng_al.integers(0, 2)) if al_exhaust is None else bool(al_exhaust)
    rot = int(rng_al.integers(0, 4)) if rotation is None else int(rotation)
    rng = np.random.default_rng(seed)
    for _ in range(400):
        lay = _try_build(rng, k, exh, rot, partition_prob)
        if lay is not None:
            lay["seed"] = seed
            return lay
    raise RuntimeError("не вдалося побудувати план для seed=%s" % seed)


def make_simple_room(sup_xy=(CX - 12, CY), ret_xy=(CX + 12, CY)):
    """Проста кругла кімната без зон і шлюзу (для перевірок симулятора)."""
    yy, xx = np.mgrid[0:N, 0:N]
    fluid = np.hypot(xx - CX, yy - CY) <= R - 1
    z = np.zeros((N, N), bool)
    supply = z.copy()
    supply[sup_xy[1], sup_xy[0]] = True
    ret = z.copy()
    ret[ret_xy[1], ret_xy[0]] = True
    lay = dict(
        fluid=fluid,
        zone=np.zeros((N, N), np.int8),
        airlock=np.zeros((N, N), np.int8),
        supply=supply,
        ret=ret,
        alexh=z.copy(),
        source=z.copy(),
        n_stages=0,
        al_exhaust=False,
        n_supply=1,
        partitions=[],
        rotation=0,
        door_cell=(CY, CX + R - 2),
        path_m={},
        area_m2=float(fluid.sum() * DX * DX),
        volume_m3=float(fluid.sum() * DX * DX * H),
    )
    return lay


def channels(lay):
    """Вхід для нейромережі: (C, N, N) float32."""
    ch = [
        lay["fluid"],
        lay["supply"],
        lay["ret"],
        lay["alexh"],
        lay["source"],
        lay["airlock"] > 0,
    ]
    ch += [lay["zone"] == zi for zi in range(1, len(ZONES) + 1)]
    return np.stack(ch).astype(np.float32)
