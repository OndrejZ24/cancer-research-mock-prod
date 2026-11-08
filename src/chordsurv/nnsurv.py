"""Neural discrete-time survival model (torch). See boosting.py: never import both in one process."""
import copy

import numpy as np
import torch
from torch import nn

from .data import SEED
from .models import N_INTERVALS, es_split, interval_targets


class HazardNet(nn.Module):
    """Multilayer perceptron with one output per month: the logit of the hazard of dying in that month."""

    def __init__(self, n_in, width, depth, dropout, n_out=N_INTERVALS):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        return self.head(self.body(x))


def nll(logits, survived, died, w):
    """Weighted mean negative log-likelihood of the discrete-time model:
    sum over months of survived * softplus(logit) + died * softplus(-logit)."""
    per_row = (survived * nn.functional.softplus(logits) + died * nn.functional.softplus(-logits)).sum(dim=1)
    return (w * per_row).sum() / w.sum()


def nn_fit(Z, time, event, weight, rows, n_patients, params, seed=SEED, max_epochs=300, patience=20, batch=512):
    """AdamW on mini-batches; stops when the loss on an inner 15 % of patients has not improved for `patience`
    epochs and returns the weights of the best epoch."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    es = es_split(n_patients, rows, seed=SEED)                # same inner split for every seed
    survived, died = interval_targets(time, event)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
    Xf, Sf, Df, Wf = T(Z[~es]), T(survived[~es]), T(died[~es]), T(weight[~es])
    Xe, Se, De, We = T(Z[es]), T(survived[es]), T(died[es]), T(weight[es])

    net = HazardNet(Z.shape[1], params["width"], params["depth"], params["dropout"])
    with torch.no_grad():                                     # start from the average hazard of each month
        h = ((Df * Wf[:, None]).sum(0) / ((Sf + Df) * Wf[:, None]).sum(0).clamp(min=1e-8)).clamp(1e-5, 1 - 1e-5)
        net.head.bias.copy_(torch.log(h / (1 - h)))
    opt = torch.optim.AdamW(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best, best_state, best_epoch, wait, curve = np.inf, None, 0, 0, []
    for epoch in range(max_epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(len(Xf)))
        for i in range(0, len(Xf), batch):
            b = perm[i:i + batch]
            loss = nll(net(Xf[b]), Sf[b], Df[b], Wf[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val, train = nll(net(Xe), Se, De, We).item(), nll(net(Xf), Sf, Df, Wf).item()
        curve.append((train, val))
        if val < best - 1e-5:
            best, best_state, best_epoch, wait = val, copy.deepcopy(net.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    net.zero_grad()
    nll(net(Xf), Sf, Df, Wf).backward()                       # full-batch gradient norm at the returned weights
    gnorm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters())))
    return {"net": net, "best_epoch": best_epoch + 1, "epochs_run": epoch + 1, "es_loss": best, "curve": curve,
            "grad_norm": gnorm, "n_params": sum(p.numel() for p in net.parameters()), "params": dict(params),
            "seed": seed}


def nn_risk(models, Z):
    """12-month risk = 1 - prod over months of (1 - hazard), averaged over the seeds."""
    X = torch.as_tensor(np.ascontiguousarray(Z), dtype=torch.float32)
    out = []
    for m in models:
        with torch.no_grad():
            out.append(1.0 - torch.exp(-nn.functional.softplus(m["net"](X)).sum(dim=1)).numpy())
    return np.mean(out, axis=0)
