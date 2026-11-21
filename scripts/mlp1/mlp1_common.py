"""mlp1 shared code: al2 data (last-token X_L*, word-level split), decoding exactly as scripts/al2/fit.py and
scripts/mp1/mp1_fit.py ((v * g) @ U.T, excl mask, rank = best over the word's single-token forms), the M3 ridge recipe
(whitener P from within-word covariance, ridge on per-word means), Jgen / Jrel lenses, GPU gate."""
import json, os, subprocess, sys, time
from collections import defaultdict
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

R = "/workspace/o-lens-improvement"
BIG = f"{R}/.hf27/al2"
BIG1 = f"{R}/.hf27/al1"
OUT = f"{R}/.hf27/mlp1"
RES = f"{R}/scripts/mlp1/results"
sys.path.insert(0, f"{R}/src")
GRID = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
EPSF = [0.01, 0.1]
READ = [4, 8, 12, 16, 20, 24, 28]
LAMS = [0.1, 0.3, 1.0]
dev = os.environ.get("MLP1_DEV", "cuda")
T00 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T00:6.0f}s]", *a, flush=True)


def gpu_setup(frac=0.11, need_gb=10.0):
    """Wait until >= need_gb free (nvidia-smi), then cap this process at frac of the GPU."""
    while True:
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.split()
            free = float(q[0]) / 1024
        except Exception:
            free = 0.0
        if free >= need_gb:
            break
        log(f"waiting for GPU memory: {free:.1f} GB free")
        time.sleep(60)
    torch.cuda.set_per_process_memory_fraction(frac)
    log(f"GPU ok ({free:.1f} GB free), fraction {frac}")


def oom_retry(fn, *a, **k):
    while True:
        try:
            return fn(*a, **k)
        except torch.cuda.OutOfMemoryError:
            log("CUDA OOM: empty cache, wait 60 s, retry")
            torch.cuda.empty_cache()
            time.sleep(60)


class Data:
    def __init__(self):
        meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
        self.meta = meta
        self.words_s, split, style, role = meta["words"], meta["split"], meta["style"], meta["role"]
        self.g = meta["g"].to(dev)
        self.U = torch.load(f"{BIG1}/U.pt").to(dev)
        if dev == "cpu":
            self.U = self.U.float()
        self.V, self.d = self.U.shape
        strs = torch.load(f"{BIG1}/meta.pt", weights_only=False)["strs"]
        self.strs = strs
        self.excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
        by_norm = defaultdict(list)
        for i, s in enumerate(strs):
            by_norm[s.strip().lower()].append(i)
        self.W_all = sorted(role)
        widx = {w: i for i, w in enumerate(self.W_all)}
        forms = [by_norm[w.lower()] for w in self.W_all]
        assert all(forms)
        mf = max(len(f) for f in forms)
        self.FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)
        self.FVALID = torch.tensor([[True] * len(f) + [False] * (mf - len(f)) for f in forms], device=dev)
        self.TW = torch.stack([meta["T"][w] for w in self.W_all]).to(dev).float()
        self.wi = torch.tensor([widx[w] for w in self.words_s], device=dev)
        self.sty = torch.tensor(style, device=dev)
        spl = {r: torch.tensor([s == r for s in split], device=dev) for r in ("train", "val", "test")}
        self.TRN = spl["train"].nonzero()[:, 0]
        self.VAL = spl["val"].nonzero()[:, 0]
        self.TE = spl["test"].nonzero()[:, 0]
        g = torch.Generator().manual_seed(123)
        self.TRS = self.TRN[torch.randperm(len(self.TRN), generator=g)[:6000].to(dev)].sort().values  # train subset
        self.te_sty = self.sty[self.TE].cpu()

    def logits(self, vecs, scale=None):
        lg = ((vecs.float() * self.g).to(self.U.dtype) @ self.U.T).float()
        if scale is not None:  # scale before masking (avoids 0 * inf in the scale's gradient)
            lg = lg * scale
        return lg.masked_fill_(self.excl, float("-inf"))

    @torch.no_grad()
    def ranks_of(self, vecs, wid, bs=256):
        out = []
        for i in range(0, vecs.shape[0], bs):
            lg = self.logits(vecs[i:i + bs])
            best = lg.gather(1, self.FORM[wid[i:i + bs]]).max(1).values
            out.append(torch.count_nonzero(lg > best[:, None], dim=1) + 1)
        return torch.cat(out)

    def load_X(self, l):
        return torch.load(f"{BIG}/X_L{l}.pt")  # CPU fp16 [N, d]

    def rows(self, X, idx):
        if X.device.type == dev:  # X kept on the GPU in fp16 (1 GB): CPU row gathers were the bottleneck
            return X[idx].float()
        return X[idx.cpu()].to(dev, non_blocking=True).float()


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def lensd(p):
    from workspace_lens.utils import load_lens_file
    return {int(l): v.to(dev).float() for l, v in
            load_lens_file(f"{R}/{p}", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN.items()}


JGEN = "runs/w1/jpp_Afull_Wtrain.pt"
JREL = "artifacts/baselines/jpp9b_prefit.pt"


# ---------------- M3 ridge (as al2/fit.py, mp1_fit.py), class stats in chunks from CPU ----------------
class Ridge:
    def __init__(self, X):
        self.mx = X.mean(0)
        Xc = (X - self.mx).double()
        s, Q = torch.linalg.eigh(Xc.T @ Xc)
        self.s, self.Q, self.Xc = s.clamp_min(0), Q, Xc
        self.scale = self.s.mean().item()

    def setY(self, Y):
        self.my = Y.mean(0)
        self.QtB = self.Q.T @ (self.Xc.T @ (Y - self.my).double())

    def W(self, c):
        return (self.Q @ (self.QtB / (self.s + c * self.scale)[:, None])).float()


@torch.no_grad()
def class_stats(D, X, idx):
    ws = D.wi[idx]
    uw, inv = torch.unique(ws, return_inverse=True)
    Mn = torch.zeros(len(uw), D.d, device=dev)
    cnt = torch.zeros(len(uw), device=dev)
    for i in range(0, len(idx), 16384):
        Mn.index_add_(0, inv[i:i + 16384], D.rows(X, idx[i:i + 16384]))
        cnt.index_add_(0, inv[i:i + 16384], torch.ones(len(inv[i:i + 16384]), device=dev))
    Mn /= cnt[:, None]
    Sw = torch.zeros(D.d, D.d, device=dev, dtype=torch.float64)
    for i in range(0, len(idx), 16384):
        Rw = D.rows(X, idx[i:i + 16384]) - Mn[inv[i:i + 16384]]
        Sw += (Rw.T @ Rw).double()
    return uw, Mn, Sw / len(idx)


def whitener(Sw, f):
    s, Q = torch.linalg.eigh(Sw)
    s = s.clamp_min(0)
    return (Q @ torch.diag((s + f * s.mean()) ** -0.5) @ Q.T).float()


@torch.no_grad()
def fit_M3(D, X, tr_idx, epsf=EPSF, grid=GRID):
    """yields cfg, dict(mx, P, Wr, my): M h = ((h - mx) @ P) @ Wr + my."""
    uw, Mn, Sw = class_stats(D, X, tr_idx)
    Tm = D.TW[uw]
    for f in epsf:
        P = whitener(Sw, f)
        r = Ridge(Mn @ P)
        r.setY(Tm)
        for c in grid:
            yield dict(f=f, c=c), dict(mx=Mn.mean(0), P=P, Wr=r.W(c), my=r.my.float())
        del r, P


def apply_M(m, x):
    return ((x - m["mx"]) @ m["P"]) @ m["Wr"] + m["my"]


def t10(r):
    return float((r <= 10).float().mean())


def t1(r):
    return float((r <= 1).float().mean())


def any7(store):
    return torch.stack([store[l] for l in READ]).min(0).values
