"""Healthy-memory detector: PCA leftover and KNN distance on 2D angular-distribution vectors.

Run with the course environment (scikit-learn). Stage "tune" sweeps one knob at a time on the
training-gear cavities against healthy revolution B tuning teeth; stage "test" applies chosen
settings once to the test gears.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
LOG_FLOOR = 1e-6


def transform(binned, total, gamma):
    share = binned/total[:, None]
    if gamma == "log":
        return np.log10(share+LOG_FLOOR)
    return share**float(gamma)


def scores(vectors, data, gamma, pca_size, k, rows):
    x = transform(data["binned"], data["total"], gamma)
    memory = x[data["case"] == "healthy_a"]
    pca = PCA(n_components=pca_size, svd_solver="full").fit(memory)
    z_memory, z = pca.transform(memory), pca.transform(x[rows])
    distance, _ = NearestNeighbors(n_neighbors=k).fit(z_memory).kneighbors(z)
    leftover = np.linalg.norm(x[rows]-pca.inverse_transform(z), axis=1)
    return {"knn": distance.mean(axis=1), "leftover": leftover}, pca.explained_variance_ratio_.sum()


def evaluate(score, positive, negative, defect_key):
    """Frame-level ROC area, and cavities caught when 1% of healthy frames are flagged."""
    y = np.r_[np.ones(positive.sum()), np.zeros(negative.sum())]
    auc = roc_auc_score(y, np.r_[score[positive], score[negative]])
    cutoff = np.quantile(score[negative], 0.99)
    keys = defect_key[positive]
    caught = {key for key in set(keys) if score[positive][keys == key].max() > cutoff}
    return auc, len(caught), len(set(keys))


def tune(config, data, out):
    lo, hi = config["tuning_segments"]
    negative = (data["case"] == "healthy_b") & (data["segment"] >= lo) & (data["segment"] <= hi)
    positive = (data["role"] == "train") & (data["label"] == 1)
    check = (data["role"] == "validation") & (data["label"] == 1)
    rows = np.flatnonzero(negative | positive | check)
    defect_key = np.char.add(np.char.add(data["case"], ":"), data["defect_id"])
    current = dict(config["defaults"])
    results = []
    for knob in ("gamma", "pca_size", "k"):
        best = {}
        for value in config["sweeps"][knob]:
            setting = {**current, knob: value}
            partial, kept = scores(None, data, setting["gamma"], setting["pca_size"], setting["k"], rows)
            for name, part in partial.items():
                full = np.full(len(data["case"]), np.nan)
                full[rows] = part
                auc, caught, total = evaluate(full, positive, negative, defect_key)
                vauc, vcaught, vtotal = evaluate(full, check, negative, defect_key)
                results.append({"knob": knob, **setting, "score": name, "variance_kept": round(float(kept), 4),
                                "tuning_auc": round(auc, 4), "tuning_caught_at_1pct": f"{caught}/{total}",
                                "validation_auc": round(vauc, 4), "validation_caught_at_1pct": f"{vcaught}/{vtotal}"})
                print(f"{knob:8s} {str(value):5s} {name:8s} AUC {auc:.4f} caught {caught}/{total} | "
                      f"val AUC {vauc:.4f} caught {vcaught}/{vtotal} | variance kept {kept:.3f}", flush=True)
                if name == "knn" or knob != "k":
                    best.setdefault(name, []).append((auc, value))
        # Carry forward the value that gives the higher KNN tuning AUC.
        current[knob] = max(best["knn"])[1]
    (out/"tuning.json").write_text(json.dumps({"chosen": current, "defaults": config["defaults"],
                                               "results": results}, indent=2)+"\n")
    print("Chosen settings:", current, flush=True)


def errors(config, data, out, settings):
    """List missed cavities and flagged healthy frames on the tuning data at several thresholds."""
    lo, hi = config["tuning_segments"]
    pitch = 360/config["tooth_count"]
    negative = (data["case"] == "healthy_b") & (data["segment"] >= lo) & (data["segment"] <= hi)
    positive = (data["role"] == "train") & (data["label"] == 1)
    rows = np.flatnonzero(negative | positive)
    cases = json.loads((PROJECT/config["source_run"]/"cases.json").read_text())
    crossing = {f"{c['case']}:{d['id']}": d["crossing_angle_deg"] for c in cases["cases"] for d in c["defects"]}
    # Flank phase: the brightest 2° bin of the matching healthy-A frame lies more than 5° off the lens axis.
    count = int(data["bin_count"])
    centre = (np.arange(count)+0.5)*config["bin_deg"]-config["half_range_deg"]
    a = data["case"] == "healthy_a"
    v, u = np.unravel_index(data["binned"][a].argmax(axis=1), (count, count))
    flank_by_index = dict(zip(data["index"][a], np.maximum(abs(centre[u]), abs(centre[v])) > 5))
    report = {}
    for setting in settings:
        partial, _ = scores(None, data, setting["gamma"], setting["pca_size"], setting["k"], rows)
        score = np.full(len(data["case"]), np.nan)
        score[rows] = partial["knn"]
        healthy = score[negative]
        name = f"gamma {setting['gamma']}, PCA {setting['pca_size']}, K {setting['k']}"
        report[name] = {}
        print(f"\n=== {name}", flush=True)
        for label, cutoff in (("strict (0 healthy frames)", healthy.max()), ("1% healthy frames", np.quantile(healthy, 0.99)),
                              ("5% healthy frames", np.quantile(healthy, 0.95))):
            keys = np.char.add(np.char.add(data["case"], ":"), data["defect_id"])
            missed = []
            for key in sorted(set(keys[positive])):
                frames = positive & (keys == key)
                peak = score[frames].max()
                if peak <= cutoff:
                    first = np.flatnonzero(frames)[0]
                    near_switch = any(flank_by_index[i] for i in data["index"][frames])
                    missed.append({"cavity": key, "kind": str(data["kind"][first]), "diameter_mm": float(data["diameter_mm"][first]),
                                   "peak_over_threshold": round(float(peak/cutoff), 3), "at_flank_switch": bool(near_switch),
                                   "crossing_deg": round(crossing[key], 3)})
            flagged = negative & (score > cutoff)
            teeth = sorted(set(data["segment"][flagged].tolist()))
            phase = (data["angle_deg"][flagged] % pitch).round(2)
            in_flank = [bool(flank_by_index[i]) for i in data["index"][flagged]]
            report[name][label] = {"threshold": float(cutoff), "missed": missed,
                                   "healthy_frames_flagged": f"{int(flagged.sum())}/{int(negative.sum())}",
                                   "healthy_teeth_flagged": f"{len(teeth)}/{hi-lo+1}", "flagged_teeth": teeth,
                                   "flagged_phase_deg": phase.tolist(), "flagged_in_flank_phase": int(sum(in_flank))}
            print(f"-- {label}: missed {len(missed)}/40 cavities | healthy frames flagged {int(flagged.sum())}/{int(negative.sum())}"
                  f" | healthy teeth flagged {len(teeth)}/{hi-lo+1} | flagged frames in flank phase {sum(in_flank)}/{len(in_flank)}", flush=True)
            for m in missed:
                print(f"     missed {m['cavity']:24s} {m['kind']:8s} Ø{m['diameter_mm']:.3f}  peak/threshold {m['peak_over_threshold']:.3f}"
                      f"  flank switch {m['at_flank_switch']}", flush=True)
            if flagged.any():
                print(f"     false-alarm phase within tooth (deg, pitch {pitch:.2f}): {sorted(set(phase.tolist()))}", flush=True)
    (out/"tuning_errors.json").write_text(json.dumps(report, indent=2)+"\n")


def run_score(score, order, n, required=None):
    """Highest level that n consecutive frames all reach; the run must touch a required frame when given."""
    best = -np.inf
    for start in range(len(order)-n+1):
        window = order[start:start+n]
        if required is not None and not required[window].any():
            continue
        best = max(best, score[window].min())
    return best


def tuning_units(config, data, setting):
    """Per-frame KNN scores, healthy tuning teeth, and tuning cavities with their frames in angle order."""
    lo, hi = config["tuning_segments"]
    negative = (data["case"] == "healthy_b") & (data["segment"] >= lo) & (data["segment"] <= hi)
    damaged_cases = sorted(set(data["case"][(data["role"] == "train")]))
    rows = np.flatnonzero(negative | np.isin(data["case"], damaged_cases) | (data["case"] == "healthy_a"))
    partial, _ = scores(None, data, setting["gamma"], setting["pca_size"], setting["k"], rows)
    score = np.full(len(data["case"]), np.nan)
    score[rows] = partial["knn"]
    teeth = {s: np.flatnonzero(negative & (data["segment"] == s))[np.argsort(data["index"][negative & (data["segment"] == s)])]
             for s in range(lo, hi+1)}
    keys = np.char.add(np.char.add(data["case"], ":"), data["defect_id"])
    cavities = {}
    for key in sorted(set(keys[(data["role"] == "train") & (data["label"] == 1)])):
        case = key.split(":")[0]
        labelled = (keys == key) & (data["label"] == 1)
        centre = data["index"][labelled]
        frames = np.flatnonzero((data["case"] == case) & (abs(data["index"]-centre.mean()) <= 8))
        cavities[key] = (frames[np.argsort(data["index"][frames])], labelled)
    return score, teeth, cavities


def rules(config, data, out):
    """Tooth-level trade-off for flagging after 1, 2, or 3 consecutive frames above the cutoff."""
    report = {}
    for setting in ({"gamma": 0.5, "pca_size": 20, "k": 3}, {"gamma": 0.5, "pca_size": 5, "k": 3}):
        score, teeth, cavities = tuning_units(config, data, setting)
        name = f"PCA {setting['pca_size']}"
        print(f"\n=== gamma 0.5, {name}, K 3 | {len(teeth)} healthy teeth, {len(cavities)} cavities", flush=True)
        for n in (1, 2, 3):
            healthy = np.sort([run_score(score, frames, n) for frames in teeth.values()])[::-1]
            damage = {key: run_score(score, frames, n, labelled) for key, (frames, labelled) in cavities.items()}
            for allowed in (0, 1, 2):
                cutoff = healthy[allowed]
                missed = sorted(key for key, value in damage.items() if value <= cutoff)
                report.setdefault(name, []).append({"in_a_row": n, "healthy_teeth_flagged": allowed, "cutoff": float(cutoff),
                                                    "missed": missed})
                print(f"{n} in a row | {allowed}/18 healthy teeth flagged | missed {len(missed)}/40: "
                      f"{', '.join(k.split(':')[1] + '(' + k.split('_')[0] + ')' for k in missed)}", flush=True)
    (out/"tuning_rules.json").write_text(json.dumps(report, indent=2)+"\n")


def tradeoff(config, data, out):
    """Full FN/FP trade-off per rule: for each count of flagged healthy teeth, the cavities missed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, size in zip(axes, (5, 20)):
        score, teeth, cavities = tuning_units(config, data, {"gamma": 0.5, "pca_size": size, "k": 3})
        for n, marker in ((1, "o"), (2, "s"), (3, "^")):
            healthy = np.sort([run_score(score, frames, n) for frames in teeth.values()])[::-1]
            damage = np.array([run_score(score, frames, n, labelled) for frames, labelled in cavities.values()])
            rows = []
            for fp in range(len(healthy)):
                cutoff = healthy[fp]
                fn = int((damage <= cutoff).sum())
                rows.append({"FP": fp, "TN": len(healthy)-fp, "FN": fn, "TP": len(damage)-fn})
            table[f"PCA {size}, {n} in a row" if n > 1 else f"PCA {size}, single frame"] = rows
            label = "single frame" if n == 1 else f"{n} in a row"
            ax.plot([r["FP"] for r in rows], [r["FN"] for r in rows], marker=marker, label=label)
        ax.set(title=f"PCA {size}", xlabel="FP: healthy teeth flagged (of 18)", xticks=range(0, 18, 2))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("FN: cavities missed (of 40)")
    axes[0].legend()
    fig.suptitle("Tuning trade-off · square root, K 3 · each point is one cutoff")
    fig.tight_layout()
    (out/"images").mkdir(exist_ok=True)
    fig.savefig(out/"images/tuning_tradeoff.png", dpi=130)
    (out/"tuning_tradeoff.json").write_text(json.dumps(table, indent=2)+"\n")
    for name, rows in table.items():
        print(f"{name:24s} FN at FP=0..8: " + " ".join(f"{r['FN']:2d}" for r in rows[:9]), flush=True)


def false_negatives(config, data, out):
    """Signal against render noise for every tuning cavity, and peak score against diameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setting = {"gamma": 0.5, "pca_size": 20, "k": 3}
    score, teeth, cavities = tuning_units(config, data, setting)
    strict = max(run_score(score, frames, 1) for frames in teeth.values())
    x = transform(data["binned"], data["total"], setting["gamma"])
    twin = {name: dict(zip(data["index"][data["case"] == name], np.flatnonzero(data["case"] == name)))
            for name in ("healthy_a", "healthy_b")}
    rows = []
    for key, (frames, labelled) in cavities.items():
        peak = frames[np.argmax(np.where(labelled[frames], score[frames], -np.inf))]
        i = data["index"][peak]
        a, b = twin["healthy_a"][i], twin["healthy_b"][i]
        signal, noise = np.linalg.norm(x[peak]-x[a]), np.linalg.norm(x[b]-x[a])
        rows.append({"cavity": key, "kind": str(data["kind"][peak]), "diameter_mm": float(data["diameter_mm"][peak]),
                     "peak_over_strict": round(float(score[peak]/strict), 3), "signal_over_noise": round(float(signal/noise), 2),
                     "light_change_pct": round(float(100*(data["total"][peak]/data["total"][a]-1)), 3),
                     "outside_window_change_pct_points": round(float(100*(data["outside_window"][peak]-data["outside_window"][a])), 3)})
    rows.sort(key=lambda r: r["peak_over_strict"])
    print(f"strict per-frame cutoff {strict:.4f}; rows sorted by peak score / cutoff", flush=True)
    print(f"{'cavity':26s}{'kind':9s}{'Ø mm':>6s}{'score/cut':>10s}{'sig/noise':>10s}{'light %':>9s}{'outside pp':>11s}", flush=True)
    for r in rows:
        print(f"{r['cavity']:26s}{r['kind']:9s}{r['diameter_mm']:6.3f}{r['peak_over_strict']:10.3f}{r['signal_over_noise']:10.2f}"
              f"{r['light_change_pct']:9.3f}{r['outside_window_change_pct_points']:11.3f}", flush=True)
    (out/"tuning_false_negatives.json").write_text(json.dumps({"strict_cutoff": strict, "cavities": rows}, indent=2)+"\n")
    (out/"images").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind, marker in (("top_edge", "o"), ("crown", "s"), ("flank", "^")):
        sel = [r for r in rows if r["kind"] == kind]
        label = kind.replace("_", " ")
        axes[0].scatter([r["diameter_mm"] for r in sel], [r["peak_over_strict"] for r in sel], marker=marker, label=label)
        axes[1].scatter([r["diameter_mm"] for r in sel], [r["signal_over_noise"] for r in sel], marker=marker, label=label)
    axes[0].axhline(1, color="k", ls="--", lw=1)
    axes[0].set(xlabel="cavity diameter (mm)", ylabel="peak KNN score / strict cutoff", title="Score against size")
    axes[1].axhline(1, color="k", ls="--", lw=1)
    axes[1].set(xlabel="cavity diameter (mm)", ylabel="damage difference / render noise", title="Signal against render noise")
    axes[0].legend()
    fig.suptitle("Tuning cavities · square root, PCA 20, K 3")
    fig.tight_layout()
    fig.savefig(out/"images/tuning_false_negatives.png", dpi=130)


def neighbour_mean(values, data, width):
    """Mean over adjacent frames of the same case: width 2 uses (i, i+1), width 3 uses (i-1, i, i+1)."""
    offsets = (0, 1) if width == 2 else (-1, 0, 1)
    position = {(c, i): r for r, (c, i) in enumerate(zip(data["case"], data["index"]))}
    result = np.empty_like(values)
    for r, (c, i) in enumerate(zip(data["case"], data["index"])):
        rows = [position[(c, i+o)] for o in offsets if (c, i+o) in position]
        result[r] = values[rows].mean(axis=0)
    return result


def f1_table_row(name, score, teeth, cavities, n):
    """Per-tooth decision with the F1-best cutoff; returns all counts and rates in one row."""
    healthy = np.array([run_score(score, frames, n) for frames in teeth.values()])
    keys = list(cavities)
    damage = np.array([run_score(score, *cavities[key][:1], n, cavities[key][1]) for key in keys])
    best = None
    for cutoff in np.r_[-np.inf, np.unique(np.r_[healthy, damage])]:
        tp, fp = int((damage > cutoff).sum()), int((healthy > cutoff).sum())
        fn, tn = len(damage)-tp, len(healthy)-fp
        f1 = 2*tp/(2*tp+fp+fn)
        if best is None or (f1, -fp) > (best["F1"], -best["FP"]):
            best = {"setting": name, "TP": tp, "FN": fn, "FP": fp, "TN": tn, "F1": round(f1, 3),
                    "missed": [keys[j].split(":")[1] for j in np.flatnonzero(damage <= cutoff)]}
    total = best["TP"]+best["FN"]+best["FP"]+best["TN"]
    best.update(accuracy=round((best["TP"]+best["TN"])/total, 3), detection_rate=round(best["TP"]/len(damage), 3),
                false_alarms_per_tooth=round(best["FP"]/len(healthy), 3),
                precision=round(best["TP"]/max(best["TP"]+best["FP"], 1), 3))
    return best


def averaging(config, data, out):
    """Frame averaging of scores or vectors, PCA 5, K 3, square root; F1-best cutoff per setting."""
    setting = {"gamma": 0.5, "pca_size": 5, "k": 3}
    base, teeth, cavities = tuning_units(config, data, setting)
    used = np.flatnonzero(~np.isnan(base))
    x = transform(data["binned"], data["total"], setting["gamma"])
    variants = {"no averaging": base, "score mean of 2": neighbour_mean(base, data, 2),
                "score mean of 3": neighbour_mean(base, data, 3)}
    for width in (2, 3):
        smooth = neighbour_mean(x, data, width)
        memory = smooth[data["case"] == "healthy_a"]
        pca = PCA(n_components=setting["pca_size"], svd_solver="full").fit(memory)
        distance, _ = NearestNeighbors(n_neighbors=setting["k"]).fit(pca.transform(memory)).kneighbors(pca.transform(smooth[used]))
        score = np.full(len(data["case"]), np.nan)
        score[used] = distance.mean(axis=1)
        variants[f"vector mean of {width}"] = score
    rows = [f1_table_row(f"{name}, {'single frame' if n == 1 else '2 in a row'}", score, teeth, cavities, n)
            for name, score in variants.items() for n in (1, 2)]
    (out/f"tuning_averaging{config.get('tag', '')}.json").write_text(json.dumps(rows, indent=2)+"\n")
    print(f"{'setting':34s} TP FN FP TN  acc   det   FA/tooth prec  F1    missed", flush=True)
    for r in rows:
        print(f"{r['setting']:34s} {r['TP']:2d} {r['FN']:2d} {r['FP']:2d} {r['TN']:2d}  {r['accuracy']:.3f} {r['detection_rate']:.3f} "
              f"{r['false_alarms_per_tooth']:.3f}    {r['precision']:.3f} {r['F1']:.3f} {', '.join(r['missed'])}", flush=True)


def unit_frames(data, roles, lo, hi):
    """Healthy-B teeth in segments lo..hi, and cavities of the given roles with their nearby frames, in angle order."""
    healthy = (data["case"] == "healthy_b") & (data["segment"] >= lo) & (data["segment"] <= hi)
    teeth = {}
    for s in range(lo, hi+1):
        frames = np.flatnonzero(healthy & (data["segment"] == s))
        teeth[s] = frames[np.argsort(data["index"][frames])]
    keys = np.char.add(np.char.add(data["case"], ":"), data["defect_id"])
    cavities = {}
    for key in sorted(set(keys[np.isin(data["role"], roles) & (data["label"] == 1)])):
        case = key.split(":")[0]
        labelled = (keys == key) & (data["label"] == 1)
        frames = np.flatnonzero((data["case"] == case) & (abs(data["index"]-data["index"][labelled].mean()) <= 8))
        cavities[key] = (frames[np.argsort(data["index"][frames])], labelled)
    return teeth, cavities


def counts(score, teeth, cavities, cutoff, data):
    healthy = {s: run_score(score, frames, 1) for s, frames in teeth.items()}
    damage = {key: run_score(score, frames, 1, labelled) for key, (frames, labelled) in cavities.items()}
    tp = int(sum(v > cutoff for v in damage.values()))
    fp = int(sum(v > cutoff for v in healthy.values()))
    fn, tn = len(damage)-tp, len(healthy)-fp
    total = tp+fn+fp+tn
    detail = {}
    for key, (_, labelled) in cavities.items():
        first = np.flatnonzero(labelled)[0]
        detail[key] = {"kind": str(data["kind"][first]), "diameter_mm": float(data["diameter_mm"][first]),
                       "score_over_cutoff": round(float(damage[key]/cutoff), 3), "caught": bool(damage[key] > cutoff)}
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "accuracy": round((tp+tn)/total, 3) if total else None,
            "detection_rate": round(tp/len(damage), 3) if damage else None,
            "false_alarms_per_tooth": round(fp/len(healthy), 3) if healthy else None,
            "precision": round(tp/(tp+fp), 3) if tp+fp else None,
            "F1": round(2*tp/(2*tp+fp+fn), 3) if tp+fp+fn else None,
            "missed": sorted(k for k, d in detail.items() if not d["caught"]),
            "flagged_teeth": sorted(int(s) for s, v in healthy.items() if v > cutoff), "cavities": detail}


def test(config, data, out):
    """Decided setup: square root, PCA 5, K 3, vector mean of 2, single frame. Cutoff frozen on tuning data first."""
    setting = {"gamma": 0.5, "pca_size": 5, "k": 3, "vector_mean": 2}
    x = neighbour_mean(transform(data["binned"], data["total"], setting["gamma"]), data, setting["vector_mean"])
    memory = x[data["case"] == "healthy_a"]
    pca = PCA(n_components=setting["pca_size"], svd_solver="full").fit(memory)
    knn = NearestNeighbors(n_neighbors=setting["k"]).fit(pca.transform(memory))
    score = np.full(len(data["case"]), np.nan)
    used = np.flatnonzero(data["case"] != "healthy_a")
    score[used] = knn.kneighbors(pca.transform(x[used]))[0].mean(axis=1)
    lo, hi = config["tuning_segments"]
    teeth, cavities = unit_frames(data, ["train"], lo, hi)
    tuning = f1_table_row("tuning", score, teeth, cavities, 1)
    healthy = np.array([run_score(score, f, 1) for f in teeth.values()])
    damage = np.array([run_score(score, f, 1, l) for f, l in cavities.values()])
    candidates = np.r_[-np.inf, np.unique(np.r_[healthy, damage])]
    cutoff = float(max(candidates, key=lambda c: (2*(damage > c).sum()/(2*(damage > c).sum()+(healthy > c).sum()+(damage <= c).sum()),
                                                  -(healthy > c).sum())))
    (out/"test_cutoff.json").write_text(json.dumps({"setting": setting, "cutoff": cutoff, "chosen_on": "tuning data",
                                                     "tuning_result": tuning}, indent=2)+"\n")
    print(f"Frozen cutoff {cutoff:.6f} (F1 on tuning: TP {tuning['TP']} FN {tuning['FN']} FP {tuning['FP']} TN {tuning['TN']})", flush=True)
    lo, hi = config["test_segments"]
    test_teeth, test_cavities = unit_frames(data, ["test"], lo, hi)
    result = counts(score, test_teeth, test_cavities, cutoff, data)
    _, val_cavities = unit_frames(data, ["validation"], lo, hi)
    validation = counts(score, {}, val_cavities, cutoff, data)
    by_group = {}
    for key, d in result["cavities"].items():
        for group in (f"Ø{d['diameter_mm']:.2f} mm", d["kind"]):
            entry = by_group.setdefault(group, [0, 0])
            entry[0] += d["caught"]
            entry[1] += 1
    (out/"test_results.json").write_text(json.dumps({"setting": setting, "cutoff": cutoff, "test": result,
                                                     "by_group_caught_of_total": by_group,
                                                     "validation_gear": validation}, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cavities"}, indent=1), flush=True)
    print("by group (caught, total):", by_group, flush=True)
    print("missed detail:", {k: result["cavities"][k] for k in result["missed"]}, flush=True)
    print(f"validation gear: caught {validation['TP']}/{validation['TP']+validation['FN']}, missed {validation['missed']}", flush=True)


def main(config_path, stage):
    config = json.loads(config_path.read_text())
    out = PROJECT/config["output"]
    with np.load(out/f"vectors{config.get('tag', '')}.npz") as stored:
        data = {key: stored[key] for key in stored.files}
    if stage == "tune":
        tune(config, data, out)
    elif stage == "errors":
        errors(config, data, out, [{"gamma": 0.5, "pca_size": 5, "k": 3}, {"gamma": 0.5, "pca_size": 20, "k": 3}])
    elif stage == "rules":
        rules(config, data, out)
    elif stage == "fn":
        false_negatives(config, data, out)
    elif stage == "tradeoff":
        tradeoff(config, data, out)
    elif stage == "average":
        averaging(config, data, out)
    elif stage == "test":
        test(config, data, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).with_name("knn_pca.json"))
    parser.add_argument("--stage", choices=["tune", "errors", "rules", "fn", "tradeoff", "average", "test"], default="tune")
    args = parser.parse_args()
    main(args.config.resolve(), args.stage)
