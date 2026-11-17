
"""Справжня U-Net (зі skip-з'єднаннями) як сурогат симулятора пилу."""

import json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

torch.set_num_threads(2)
torch.manual_seed(0)
np.random.seed(0)


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1),
        nn.BatchNorm2d(o),
        nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1),
        nn.BatchNorm2d(o),
        nn.ReLU(True),
    )


class UNet(nn.Module):
    def __init__(s, cin=12, b=16):
        super().__init__()
        s.e1, s.e2, s.e3, s.bt = (
            block(cin, b),
            block(b, 2 * b),
            block(2 * b, 4 * b),
            block(4 * b, 8 * b),
        )
        s.u3, s.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), block(8 * b, 4 * b)
        s.u2, s.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), block(4 * b, 2 * b)
        s.u1, s.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), block(2 * b, b)
        s.out = nn.Conv2d(b, 1, 1)

    def forward(s, x):
        e1 = s.e1(x)
        e2 = s.e2(Fn.max_pool2d(e1, 2))
        e3 = s.e3(Fn.max_pool2d(e2, 2))
        m = s.bt(Fn.max_pool2d(e3, 2))
        d3 = s.d3(torch.cat([s.u3(m), e3], 1))
        d2 = s.d2(torch.cat([s.u2(d3), e2], 1))
        d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return s.out(d1)[:, 0]


if __name__ == "__main__":
    d = np.load("out/dataset.npz")
    X, Y, F, K, ZM = d["X"], d["Y"], d["F"], d["K"], d["ZM"]
    n = len(X)
    perm = np.random.default_rng(1).permutation(n)
    ntr, nva = int(0.7 * n), int(0.15 * n)
    tr, va, te = perm[:ntr], perm[ntr : ntr + nva], perm[ntr + nva :]
    MU, SD = float(Y[F].mean()), float(Y[F].std())
    Xt = torch.tensor(X)
    Yt = torch.tensor((Y - MU) / SD)
    Ft = torch.tensor(F).float()
    net = UNet()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    EP = 40
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 4e-3, total_steps=EP * ((len(tr) + 15) // 16)
    )

    def aug(x, y, f):
        k = np.random.randint(4)
        x, y, f = [torch.rot90(t, k, (-2, -1)) for t in (x, y, f)]
        if np.random.rand() < 0.5:
            x, y, f = [torch.flip(t, (-1,)) for t in (x, y, f)]
        return x, y, f

    def loss_fn(p, y, f):
        return (((p - y) ** 2) * f).sum() / f.sum()

    def predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                out.append(net(Xt[idx[i : i + 64]]))
        return torch.cat(out)

    best, t0, hist = 1e9, time.time(), []
    for ep in range(EP):
        net.train()
        np.random.shuffle(tr)
        tl = 0
        for i in range(0, len(tr), 16):
            b = tr[i : i + 16]
            x, y, f = aug(Xt[b], Yt[b], Ft[b])
            l = loss_fn(net(x), y, f)
            opt.zero_grad()
            l.backward()
            opt.step()
            sched.step()
            tl += l.item() * len(b)
        vl = loss_fn(predict(va), Yt[va], Ft[va]).item()
        hist.append((ep + 1, tl / len(tr), vl))
        if vl < best:
            best = vl
            torch.save(net.state_dict(), "out/unet.pt")
        print(
            "ep %2d train %.4f val %.4f  %.0fs"
            % (ep + 1, tl / len(tr), vl, time.time() - t0),
            flush=True,
        )

    net.load_state_dict(torch.load("out/unet.pt"))

    def metrics(idx):
        p = predict(idx).numpy() * SD + MU
        y = Y[idx]
        f = F[idx]
        err = (p - y)[f]
        r2 = 1 - (err**2).sum() / ((y[f] - y[f].mean()) ** 2).sum()
        zone = X[idx][:, 6:11] > 0.5
        pz = np.stack(
            [
                np.array([(10 ** p[i][zone[i, z]]).mean() for z in range(5)])
                for i in range(len(idx))
            ]
        )
        tz = ZM[idx]
        rel = np.abs(pz - tz) / tz
        ratio = np.maximum(pz, tz) / np.minimum(pz, tz)
        best_ok = float((pz.argmin(1) == tz.argmin(1)).mean())
        return dict(
            R2_log=float(r2),
            RMSE_log10=float(np.sqrt((err**2).mean())),
            MAE_log10=float(np.abs(err).mean()),
            zone_rel_err_median=float(np.median(rel)),
            zone_rel_err_p90=float(np.percentile(rel, 90)),
            zone_factor_median=float(np.median(ratio)),
            cleanest_zone_correct=best_ok,
        )

    res = dict(
        val=metrics(va),
        test=metrics(te),
        n=dict(train=len(tr), val=len(va), test=len(te)),
        MU=MU,
        SD=SD,
    )
    lowK = te[K[te] < 0.06]
    if len(lowK):
        res["test_lowK_lt_0.06"] = metrics(lowK)
    json.dump(res, open("out/metrics.json", "w"), ensure_ascii=False, indent=1)
    json.dump(hist, open("out/train_history.json", "w"))
    np.save("out/test_idx.npy", te)
    print(json.dumps(res, ensure_ascii=False, indent=1))
