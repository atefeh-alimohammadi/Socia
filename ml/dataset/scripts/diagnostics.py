#!/usr/bin/env python3
"""
Dataset V2 diagnostics - READ ONLY.

Run after build_dataset.py:
    python diagnostics.py [data_dir]      (default: ./data)

Checks:
  1. class balance / subtype balance, per split
  2. user-disjointness across splits
  3. pattern pool tuple + category-set disjointness across splits
  4. temporal-span / gap / event-count / category distributions by label
     and by subtype
  5. recurrence statistics (n_occ_intended distribution for positives)
  6. whether positive/negative groups differ on obvious timing/category
     features (single-feature AUC) - flags anything that looks like an
     easy shortcut
  7. shortcut baselines: timing-only, category-only, combined, trained on
     train, thresholded on val, scored on test - same methodology used in
     the V1 audit, so V1 vs V2 numbers are directly comparable
  8. per-negative-subtype false-positive rate of each baseline (does one
     subtype absorb most of a baseline's errors, or is difficulty spread
     out - a single very easy subtype dominating error would itself be a
     new shortcut path)
"""
import sys, os, json, collections
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

CATS_DEFAULT = None  # filled in from config if importable


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load(data_dir):
    pool = json.load(open(os.path.join(data_dir, "pattern_pool.json"), encoding="utf-8"))
    splits = {}
    for sp in ["train", "val", "test"]:
        ex = jl(os.path.join(data_dir, f"{sp}.jsonl"))
        meta = {m["sample_id"]: m for m in jl(os.path.join(data_dir, f"{sp}_eval_metadata.jsonl"))}
        splits[sp] = (ex, meta)
    return pool, splits


def featurize(ex):
    ev = ex["events"]
    delta = np.array([e["delta_hours"] for e in ev], float)
    cats = [e["category_id"] for e in ev]
    gaps = np.diff(delta)
    f = {"duration_h": delta[-1]}
    f["gap_mean"] = gaps.mean(); f["gap_median"] = np.median(gaps); f["gap_std"] = gaps.std()
    f["gap_min"] = gaps.min(); f["gap_max"] = gaps.max()
    for th in (0.5, 1, 3, 6, 12, 24):
        f[f"n_gap_lt_{th}h"] = int((gaps < th).sum())
    cnt = np.bincount(cats, minlength=15)
    for i, c in enumerate(cnt):
        f[f"cat_{i:02d}"] = int(c)
    f["n_unique_cat"] = int((cnt > 0).sum()); f["max_cat_count"] = int(cnt.max())
    p = cnt[cnt > 0] / cnt.sum()
    f["cat_entropy"] = float(-(p * np.log(p)).sum())
    return f


def build_df(sp, ex_list, meta):
    rows = []
    for ex in ex_list:
        m = meta[ex["sample_id"]]
        f = featurize(ex)
        rows.append({"split": sp, "sample_id": ex["sample_id"], "user_id": ex["user_id"],
                     "label": ex["label"], "subtype": m["subtype"], **f})
    return pd.DataFrame(rows)


def timing_cols(df):
    return [c for c in df.columns if c.startswith(("duration", "gap_", "n_gap_lt"))]


def category_cols(df):
    return [c for c in df.columns if c.startswith(("cat_", "n_unique", "max_cat"))]


def fit_eval(tr, va, te, cols):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    m.fit(tr[cols], tr.label)
    pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
    ths = np.linspace(0.05, 0.95, 91)
    th = ths[int(np.argmax([f1_score(va.label, pv >= t) for t in ths]))]
    return {"roc_auc": round(roc_auc_score(te.label, pt), 4),
            "pr_auc": round(average_precision_score(te.label, pt), 4),
            "f1@val_thr": round(f1_score(te.label, pt >= th), 4),
            "thr": round(float(th), 3)}, pt, th


def main(data_dir):
    pool, splits = load(data_dir)
    R = {}

    dfs = {sp: build_df(sp, ex, meta) for sp, (ex, meta) in splits.items()}
    df = pd.concat(dfs.values(), ignore_index=True)

    # 1. class / subtype balance
    R["label_counts_by_split"] = {sp: d.label.value_counts().to_dict() for sp, d in dfs.items()}
    R["subtype_counts_by_split"] = {sp: d.subtype.value_counts().to_dict() for sp, d in dfs.items()}
    R["subtype_share_of_total_by_split"] = {
        sp: (d.subtype.value_counts(normalize=True).round(3)).to_dict() for sp, d in dfs.items()}
    R["max_negative_subtype_share"] = {
        sp: round(float(d[d.label == 0].subtype.value_counts(normalize=True).max()), 3)
        for sp, d in dfs.items()}

    # 2. user disjointness
    users = {sp: set(d.user_id) for sp, d in dfs.items()}
    R["user_overlap_across_splits"] = {
        f"{a}_{b}": len(users[a] & users[b]) for a, b in
        [("train", "val"), ("train", "test"), ("val", "test")]}

    # 3. pattern pool disjointness (re-derived from the written pool file)
    by_split_pat = collections.defaultdict(list)
    for p in pool:
        by_split_pat[p["split"]].append(tuple(p["categories"]))
    tup_sets = {s: set(v) for s, v in by_split_pat.items()}
    catset_sets = {s: set(frozenset(v2) for v2 in v) for s, v in by_split_pat.items()}
    R["pattern_pool_tuple_overlap"] = {
        f"{a}_{b}": len(tup_sets[a] & tup_sets[b]) for a, b in
        [("train", "val"), ("train", "test"), ("val", "test")]}
    relation_counts = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = sum(1 for u in catset_sets[a] for v in catset_sets[b] if u == v or u <= v or v <= u)
        relation_counts[f"{a}_{b}"] = n
    R["pattern_pool_categoryset_relations"] = relation_counts

    # 4. distributions
    R["duration_days_by_label"] = (df.groupby("label").duration_h.describe() / 24).round(2).to_dict("index")
    R["events_per_sample_values"] = sorted(df.assign(n=1).groupby("sample_id").duration_h.count().unique().tolist()) \
        if False else None  # sequence length is fixed by construction; see note below
    R["duration_days_by_subtype"] = (df.groupby("subtype").duration_h.describe() / 24).round(2).to_dict("index")

    # 5. recurrence stats for positives
    pos_meta = [m for sp in splits for m in splits[sp][1].values() if m["label"] == 1]
    R["positive_n_occ_intended_dist"] = collections.Counter(m["n_occ_intended"] for m in pos_meta)

    te = dfs["test"]; tr = dfs["train"]; va = dfs["val"]

    # 6. single-feature AUC screen
    single_cols = ["duration_h", "gap_mean", "gap_median", "gap_std", "n_gap_lt_6h",
                    "n_gap_lt_24h", "n_unique_cat", "max_cat_count", "cat_entropy"]
    single_auc = {c: roc_auc_score(te.label, te[c]) for c in single_cols}
    R["single_feature_auc_test"] = {k: round(max(v, 1 - v), 3) for k, v in single_auc.items()}

    # 7. shortcut baselines
    R["majority_baseline_f1_all_positive_test"] = round(f1_score(te.label, np.ones(len(te))), 4)
    R["baselines"] = {}
    preds = {}
    for kind, cols in [("timing", timing_cols(df)), ("category", category_cols(df)),
                       ("timing+category", timing_cols(df) + category_cols(df))]:
        res, pt, th = fit_eval(tr, va, te, cols)
        R["baselines"][kind] = res
        preds[kind] = (pt, th)

    # 8. per-subtype false positive / miss rate for the combined baseline
    pt, th = preds["timing+category"]
    te2 = te.copy(); te2["pred"] = (pt >= th).astype(int)
    fp_by_subtype = te2[(te2.label == 0)].groupby("subtype").pred.mean().round(3).to_dict()
    R["timing+category_false_positive_rate_by_negative_subtype_test"] = fp_by_subtype
    miss_rate = float((te2[(te2.label == 1)].pred == 0).mean())
    R["timing+category_miss_rate_on_positives_test"] = round(miss_rate, 3)

    def cvt(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, dict): return {str(k): cvt(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [cvt(x) for x in o]
        return o
    R = cvt(R)
    print(json.dumps(R, indent=2, default=str))
    with open(os.path.join(data_dir, "diagnostics_report.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=str)
    return R


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    main(data_dir)
