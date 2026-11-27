"""q1 D3: what does the question add? Compare the two fitted A-lens maps (topic E_t3, unstated word E_s5) per layer.
Maps: out(h) = (h - mx) W + my (.hf27/ask3/E_<tag>_t3.pt, .hf27/ask5/E_<tag>_s5.pt). "Constant removed" = W only.
Per readout layer:
  cos_W           Frobenius cosine of W_topic and W_unstated
  angles_direct   principal-angle cosines between the top-50 INPUT subspaces (left singular vectors of W, d_in side)
  angles_logit    same for the effective logit map W diag(g) Wu^T (g = final-norm gain, Wu = LM head): top-50
                  eigenvectors of W diag(g) (Wu^T Wu) diag(g) W^T
  var_shared      on the benchmark (fit + dev) + probe activations: fraction of the centred output variance of one map
                  inside the top-k principal output directions of the other (k = 10, 50), both ways
  item_cos        per-item cosine of the two centred outputs (mean)
  top10_overlap   per item, |top10(map1) & top10(map2)| (f4 word groups), full maps (with constant) and W only
  supply          per row and layer: is the concept a top-10 hit of the averaged A-lens, of the topic map alone,
                  of the unstated map alone
    python scripts/q1/q1_d3.py [model]     -> runs/q1/d3_<tag>.json
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/q1")
import q1_lib2 as Q  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
TAG = Q.TAGS[MODEL]
K = 50


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def top_eig(M, k):
    s, U = torch.linalg.eigh(M.double())
    return U[:, -k:].float()


def angles(U1, U2):
    return torch.linalg.svdvals(U1.T @ U2).tolist()


def main():
    t0 = time.time()
    model, cfg, bench, FL, F = Q.setup(MODEL)
    tok = model.tokenizer
    rows, qd = Q.load_items(model, bench, TAG)
    spare = {t for r in rows if r["set"] == "probes" for t in r["ids"]}
    S = Q.Scorer(F, spare)
    layers = list(cfg["layers"])
    dev = "cuda"
    import alens
    E = alens.load_maps(f"{Q.R}/.hf27/ask3/E_{TAG}_t3.pt", dev)
    E2 = alens.load_maps(f"{Q.R}/.hf27/ask5/E_{TAG}_s5.pt", dev)
    Wu = model._lm_head.weight.float()
    G = torch.zeros(Wu.shape[1], Wu.shape[1], device=Wu.device, dtype=torch.float64)
    for a in range(0, Wu.shape[0], 16384):
        c = Wu[a:a + 16384].double()
        G += c.T @ c
    d = model.d_model
    g = model._final_norm(torch.ones(1, d, device=dev, dtype=torch.bfloat16)).float()[0]  # gain on a unit-rms input
    Gg = (g[:, None] * G.float().to(dev) * g[None, :])
    # activations: raw texts (recomputed, last token) for every row's text
    texts = list(dict.fromkeys(r["text"] for r in rows))
    X = {l: [] for l in layers}
    for t in texts:
        st, _, _ = Q.raw_states(model, t, layers)
        for l in layers:
            X[l].append(st[l])
    X = {l: torch.stack(v) for l, v in X.items()}
    tix = {t: i for i, t in enumerate(texts)}
    out = dict(model=MODEL, layers=layers, n_texts=len(texts), per_layer={}, rows={})
    for l in layers:
        mx1, W1, my1 = E[l]
        mx2, W2, my2 = E2[l]
        res = dict(cos_W=float((W1 * W2).sum() / (W1.norm() * W2.norm())))
        U1 = torch.linalg.svd(W1, full_matrices=False)[0][:, :K]
        U2 = torch.linalg.svd(W2, full_matrices=False)[0][:, :K]
        a = angles(U1, U2)
        res["angles_direct"] = dict(top5=a[:5], mean_cos2=sum(x * x for x in a) / K, random_mean_cos2=K / d)
        V1 = top_eig(W1 @ Gg @ W1.T, K)
        V2 = top_eig(W2 @ Gg @ W2.T, K)
        a = angles(V1, V2)
        res["angles_logit"] = dict(top5=a[:5], mean_cos2=sum(x * x for x in a) / K)
        # cross: direct input subspace of one vs logit input subspace of itself (how much the unembedding reshapes)
        res["direct_vs_logit_self"] = [sum(x * x for x in angles(U1, V1)) / K, sum(x * x for x in angles(U2, V2)) / K]
        Xl = X[l]
        Y1 = (Xl - mx1) @ W1
        Y2 = (Xl - mx2) @ W2
        Y1c, Y2c = Y1 - Y1.mean(0), Y2 - Y2.mean(0)
        vs = {}
        for k in (10, 50):
            P2 = torch.linalg.svd(Y2c, full_matrices=False)[2][:k]  # [k, d] principal output directions
            P1 = torch.linalg.svd(Y1c, full_matrices=False)[2][:k]
            vs[f"topic_in_unstated_top{k}"] = float((Y1c @ P2.T).pow(2).sum() / Y1c.pow(2).sum())
            vs[f"unstated_in_topic_top{k}"] = float((Y2c @ P1.T).pow(2).sum() / Y2c.pow(2).sum())
            vs[f"topic_in_own_top{k}"] = float((Y1c @ P1.T).pow(2).sum() / Y1c.pow(2).sum())
            vs[f"unstated_in_own_top{k}"] = float((Y2c @ P2.T).pow(2).sum() / Y2c.pow(2).sum())
        res["var_shared"] = vs
        res["item_cos"] = float(torch.nn.functional.cosine_similarity(Y1c, Y2c, dim=1).mean())
        res["item_cos_full"] = float(torch.nn.functional.cosine_similarity(Y1 + my1, Y2 + my2, dim=1).mean())
        # constant share: how big is my relative to the per-item part
        res["const_share"] = [float(my1.pow(2).sum() / (Y1 + my1).pow(2).sum(1).mean()),
                              float(my2.pow(2).sum() / (Y2 + my2).pow(2).sum(1).mean())]
        # outputs: full maps (as in the A-lens), W only, and the A-lens average
        outs = dict(topic=unit(Y1 + my1), unstated=unit(Y2 + my2), alens=unit(unit(Y1 + my1) + unit(Y2 + my2)),
                    topic_W=unit(Y1c), unstated_W=unit(Y2c))
        lg = {k: S.masked(model.unembed(v).float()) for k, v in outs.items()}
        tops = {k: [S.top(v[i]) for i in range(len(texts))] for k, v in lg.items()}
        gid = S.gid

        def ov(a, b):
            return sum(len({int(gid[x]) for x in p} & {int(gid[x]) for x in q}) for p, q in zip(a, b)) / len(a)

        res["top10_overlap"] = dict(full=ov(tops["topic"], tops["unstated"]), W_only=ov(tops["topic_W"], tops["unstated_W"]),
                                    alens_vs_topic=ov(tops["alens"], tops["topic"]),
                                    alens_vs_unstated=ov(tops["alens"], tops["unstated"]))
        res["examples"] = {texts[i][-80:]: {k: [tok.decode([x]) for x in tops[k][i]] for k in ("topic", "unstated", "alens",
                                                                                               "topic_W", "unstated_W")}
                           for i in range(0, len(texts), 40)}
        for r in rows:
            i = tix[r["text"]]
            rr = out["rows"].setdefault(r["id"], dict(set=r["set"], task=r["task"], concept=r["concept"], ranks={}))
            rr["ranks"][l] = {k: S.rank(lg[k][i], r["ids"]) for k in ("topic", "unstated", "alens", "topic_W", "unstated_W")}
        out["per_layer"][l] = res
        print(f"layer {l}: {json.dumps({k: v for k, v in res.items() if k != 'examples'})} {time.time() - t0:.0f}s",
              flush=True)
    json.dump(out, open(f"{Q.R}/runs/q1/d3_{TAG}.json", "w"), ensure_ascii=False)
    print(f"wrote runs/q1/d3_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
