#!/usr/bin/env python3
"""
analyze_training.py — Parse StyleTTS2 training logs (Stage-1/Stage-2),
estimate convergence vs plateau/divergence, and report:
- best epoch so far & time since last improvement (epochs + wall-clock if timestamps exist)
- train/val trends (mean, EMA, slope, Δ%)
- generalization gap trend
- GAN stability snapshot (gen/disc oscillation & correlation)
- projections to end-of-run and next horizon

Usage (auto-derives total epochs from logs):
  python analyze_training.py --logs /path/to/train_first.log

Multiple logs (merged analysis):
  python analyze_training.py --logs /p/train_first.log /p/train_second.log

Options:
  --window 10      # epochs window for trend (default: 20% of parsed epochs, min 6)
  --horizon 10     # projection horizon in epochs (default: 10)
  --out_dir out    # where to write JSON/MD (default: alongside first log)
  --plots          # write a simple loss_curves.png if matplotlib is available
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Optional deps (script works without them)
try:
    import numpy as np
except Exception:
    np = None

# ===================== Regexes =====================

TIMESTAMP_RE = re.compile(
    r'(?P<level>DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)?[:\s]?(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,\.]\d{3})?)'
)

STEP_RE = re.compile(
    r'Epoch\s*\[(?P<epoch>\d+)\s*/\s*(?P<epoch_total>\d+)\]\s*,?\s*'
    r'Step\s*\[(?P<step>\d+)\s*/\s*(?P<step_total>\d+)\]\s*,?\s*'
    r'(?P<rest>.*)'
)

LOSS_KEYS = [
    ('mel', r'Mel\s*Loss'),
    ('gen', r'Gen\s*Loss'),
    ('disc', r'Disc\s*Loss'),
    ('slm', r'SLM\s*Loss'),
    ('s2s', r'S2S\s*Loss'),
    ('style', r'Style\s*Loss'),
    ('diff', r'Diff(?:usion)?\s*Loss'),
    ('dur', r'Dur(?:ation)?\s*Loss'),
    ('ce', r'CE\s*Loss'),
]
NUM_RE = r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'

VAL_RE_LIST = [
    re.compile(r'Validation[^0-9]*Mel[^0-9]*' + NUM_RE, re.IGNORECASE),
    re.compile(r'Val[^0-9]*Mel[^0-9]*' + NUM_RE, re.IGNORECASE),
    re.compile(r'Validation[^0-9]*loss[^0-9]*' + NUM_RE, re.IGNORECASE),
]

PHASE_MARKERS = [
    re.compile(r'entering\s+diffusion', re.IGNORECASE),
    re.compile(r'entering\s+joint', re.IGNORECASE),
]

# ===================== Data types =====================

@dataclass
class StepRecord:
    epoch: int
    step: int
    mel: Optional[float] = None
    gen: Optional[float] = None
    disc: Optional[float] = None
    slm: Optional[float] = None
    s2s: Optional[float] = None
    style: Optional[float] = None
    diff: Optional[float] = None
    dur: Optional[float] = None
    ce: Optional[float] = None
    lr: Optional[float] = None
    ts: Optional[float] = None  # POSIX timestamp

@dataclass
class ValRecord:
    epoch: int
    val_mel: float
    ts: Optional[float] = None

# ===================== Helpers =====================

def _parse_timestamp(line: str) -> Optional[float]:
    m = TIMESTAMP_RE.search(line)
    if not m:
        return None
    ts_raw = m.group('ts').replace(',', '.')
    from datetime import datetime
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(ts_raw, fmt).timestamp()
        except ValueError:
            pass
    return None

def _extract_losses(rest: str) -> Dict[str, float]:
    out = {}
    m_lr = re.search(r'LR[:\s]+([0-9eE\.\+\-]+)', rest)
    if m_lr:
        try:
            out['lr'] = float(m_lr.group(1))
        except Exception:
            pass
    for key, pat in LOSS_KEYS:
        m = re.search(pat + r'[:\s]+' + NUM_RE, rest, flags=re.IGNORECASE)
        if m:
            try:
                out[key] = float(m.group(1))
            except Exception:
                pass
    return out

def parse_log_file(path: Path):
    steps: List[StepRecord] = []
    vals: List[ValRecord] = []
    phases: List[Tuple[int, str]] = []
    epoch_total_candidates: List[int] = []

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            ts = _parse_timestamp(line)

            m = STEP_RE.search(line)
            if m:
                ep = int(m.group('epoch'))
                st = int(m.group('step'))
                try:
                    ep_total = int(m.group('epoch_total'))
                    epoch_total_candidates.append(ep_total)
                except Exception:
                    pass
                rest = m.group('rest')
                losses = _extract_losses(rest)
                rec = StepRecord(
                    epoch=ep, step=st, ts=ts,
                    mel=losses.get('mel'), gen=losses.get('gen'), disc=losses.get('disc'),
                    slm=losses.get('slm'), s2s=losses.get('s2s'),
                    style=losses.get('style'), diff=losses.get('diff'),
                    dur=losses.get('dur'), ce=losses.get('ce'),
                    lr=losses.get('lr'))
                steps.append(rec)
                continue

            # validation lines
            for val_re in VAL_RE_LIST:
                mv = val_re.search(line)
                if mv:
                    try:
                        v = float(mv.group(1))
                    except Exception:
                        continue
                    mep = re.search(r'Epoch\s*\[(\d+)', line, flags=re.IGNORECASE)
                    ep = int(mep.group(1)) if mep else (steps[-1].epoch if steps else 0)
                    vals.append(ValRecord(epoch=ep, val_mel=v, ts=ts))
                    break

            # phase markers
            for pm in PHASE_MARKERS:
                if pm.search(line):
                    ep = steps[-1].epoch if steps else 0
                    phases.append((ep, pm.pattern))
                    break

    return steps, vals, phases, epoch_total_candidates

def _np_polyfit(x, y):
    if np is None or len(x) < 2:
        return 0.0, float(y[-1]) if y else 0.0
    try:
        coeff = np.polyfit(x, y, deg=1)
        return float(coeff[0]), float(coeff[1])  # slope, intercept
    except Exception:
        return 0.0, float(y[-1]) if y else 0.0

def _ema(values: List[float], alpha: float = 0.2) -> float:
    if not values:
        return float('nan')
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(ema)

def _mean(values: List[float]) -> float:
    if not values:
        return float('nan')
    return float(sum(values) / len(values))

def _std(values: List[float]) -> float:
    if not values:
        return float('nan')
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / max(1, len(values) - 1)
    return float(var ** 0.5)

def _corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float('nan')
    mx, my = _mean(x), _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    if den == 0:
        return float('nan')
    return float(num / den)

# ===================== Core analysis =====================

def summarize(steps: List[StepRecord],
              vals: List[ValRecord],
              total_epochs_arg: Optional[int],
              epoch_total_candidates: List[int],
              window: Optional[int],
              horizon: int):
    # derive total epochs if not provided
    if total_epochs_arg is not None:
        total_epochs = total_epochs_arg
    elif epoch_total_candidates:
        total_epochs = max(epoch_total_candidates)
    else:
        # fallback: use max epoch seen
        total_epochs = max([s.epoch for s in steps] + [v.epoch for v in vals]) if (steps or vals) else None

    # Aggregate per epoch
    epochs = sorted({s.epoch for s in steps} | {v.epoch for v in vals})
    if not epochs:
        raise SystemExit("No epochs found in logs.")
    per_epoch: Dict[int, Dict[str, float]] = {}
    for e in epochs:
        e_steps = [s for s in steps if s.epoch == e]
        train_mel = _mean([s.mel for s in e_steps if s.mel is not None])
        train_gen = _mean([s.gen for s in e_steps if s.gen is not None])
        train_disc = _mean([s.disc for s in e_steps if s.disc is not None])
        v_lines = [v.val_mel for v in vals if v.epoch == e]
        val_mel = _mean(v_lines) if v_lines else float('nan')
        per_epoch[e] = dict(train_mel=train_mel, train_gen=train_gen, train_disc=train_disc, val_mel=val_mel)

    ep_sorted = sorted(per_epoch.keys())
    cur_epoch = ep_sorted[-1]

    # Window default: 20% of epochs parsed, min 6
    if window is None:
        window = max(6, int(0.2 * len(ep_sorted)))
    window = min(window, len(ep_sorted))

    def last_k(series_name: str):
        ys = [per_epoch[e][series_name] for e in ep_sorted if not math.isnan(per_epoch[e][series_name])]
        es = [e for e in ep_sorted if not math.isnan(per_epoch[e][series_name])]
        if not ys:
            return [], [], float('nan'), float('nan'), float('nan'), 'n/a', float('nan')
        k = min(window, len(es))
        es_k, ys_k = es[-k:], ys[-k:]
        mean_k = _mean(ys_k)
        ema_k = _ema(ys_k, alpha=0.2)
        slope, intercept = _np_polyfit(es_k, ys_k)
        # relative improvement over last half of window
        h = max(2, k // 2)
        early = _mean(ys_k[:k - h])
        late = _mean(ys_k[k - h:])
        rel_delta = (early - late) / early if (early not in (None, 0.0) and not math.isnan(early)) else float('nan')
        # verdict
        eps = 0.001 * (np.median(ys_k) if np is not None else (mean_k if not math.isnan(mean_k) else 1.0))
        if slope < -eps or (not math.isnan(rel_delta) and rel_delta <= -0.02):
            verdict = "improving"
        elif abs(slope) <= eps and (math.isnan(rel_delta) or rel_delta > -0.01):
            verdict = "plateau"
        else:
            verdict = "worsening"
        return es_k, ys_k, mean_k, ema_k, slope, verdict, rel_delta

    # Series: train/val
    es_tm, ys_tm, mean_tm, ema_tm, slope_tm, verdict_tm, delta_tm = last_k('train_mel')
    es_vm, ys_vm, mean_vm, ema_vm, slope_vm, verdict_vm, delta_vm = last_k('val_mel')

    # Gap
    gap_series, gap_epochs = [], []
    for e in ep_sorted:
        tm = per_epoch[e]['train_mel']; vm = per_epoch[e]['val_mel']
        if not math.isnan(tm) and not math.isnan(vm):
            gap_series.append(vm - tm)
            gap_epochs.append(e)
    if gap_series:
        k = min(window, len(gap_epochs))
        ge_k, gy_k = gap_epochs[-k:], gap_series[-k:]
        gap_mean = _mean(gy_k)
        gap_ema = _ema(gy_k, 0.2)
        gap_slope, _ = _np_polyfit(ge_k, gy_k)
        eps_gap = 0.001 * (np.median(gy_k) if np is not None else (gap_mean if not math.isnan(gap_mean) else 1.0))
        gap_verdict = "widening" if gap_slope > eps_gap else ("shrinking" if gap_slope < -eps_gap else "flat")
    else:
        gap_mean = gap_ema = gap_slope = float('nan')
        gap_verdict = "n/a"

    # GAN stability based on recent steps in window
    recent_epochs = set(es_tm or ep_sorted[-window:])
    recent_steps = [s for s in steps if s.epoch in recent_epochs]
    gen_vals = [s.gen for s in recent_steps if s.gen is not None]
    disc_vals = [s.disc for s in recent_steps if s.disc is not None]
    def _std_safe(vs): return _std(vs) if vs else float('nan')
    gen_std, disc_std = _std_safe(gen_vals), _std_safe(disc_vals)
    corr_gd = _corr(gen_vals, disc_vals) if gen_vals and disc_vals else float('nan')
    gan_verdict = "stable"
    if not math.isnan(corr_gd) and (corr_gd > -0.1 or corr_gd < -0.8):
        gan_verdict = "watch"
    if (not math.isnan(gen_std) and gen_std < 1e-6) or (not math.isnan(disc_std) and disc_std < 1e-6):
        gan_verdict = "suspect"

    # Best epoch + time since best (epochs + wall-clock)
    best_epoch, best_val, best_ts = None, float('inf'), None
    for v in vals:
        if not math.isnan(v.val_mel) and v.val_mel < best_val:
            best_val, best_epoch, best_ts = v.val_mel, v.epoch, v.ts
    since_best_epochs = (cur_epoch - best_epoch) if best_epoch is not None else None
    # Wall-clock since best: last timestamp in log minus best_ts
    last_ts = None
    for seq in (vals[::-1], steps[::-1]):
        for rec in seq:
            if getattr(rec, 'ts', None):
                last_ts = rec.ts
                break
        if last_ts:
            break
    since_best_time = None
    if best_ts and last_ts:
        delta_sec = max(0.0, last_ts - best_ts)
        # format to human readable string
        hrs = int(delta_sec // 3600); mins = int((delta_sec % 3600)//60)
        since_best_time = f"{hrs}h {mins}m"

    # Projection for val_mel
    def _project(es, ys, E):
        if not es or not ys:
            return float('nan'), 0.0
        slope, intercept = _np_polyfit(es, ys)
        # naive R^2
        yhat = [slope*x + intercept for x in es]
        mean_y = _mean(ys)
        ss_res = sum((a-b)**2 for a,b in zip(ys, yhat))
        ss_tot = sum((a-mean_y)**2 for a in ys)
        r2 = 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0
        return slope*E + intercept, r2

    proj_to = total_epochs if total_epochs else cur_epoch
    proj_end, r2_end = _project(es_vm, ys_vm, proj_to)
    proj_next, r2_next = _project(es_vm, ys_vm, cur_epoch + horizon)
    current_val = ys_vm[-1] if ys_vm else float('nan')
    delta_to_end = (proj_end - current_val) if not math.isnan(current_val) and not math.isnan(proj_end) else float('nan')
    delta_next = (proj_next - current_val) if not math.isnan(current_val) and not math.isnan(proj_next) else float('nan')

    # Overall status
    eps = 0.001 * (np.median(ys_vm) if (np is not None and ys_vm) else (mean_vm if not math.isnan(mean_vm) else 1.0))
    if ys_vm:
        if slope_vm < -eps or (not math.isnan(delta_vm) and delta_vm <= -0.02):
            status = "CONVERGING"
        elif abs(slope_vm) <= eps and (math.isnan(delta_vm) or delta_vm > -0.01):
            status = "PLATEAU"
        else:
            status = "DIVERGING"
    else:
        status = "UNKNOWN"

    summary = {
        "current_epoch": cur_epoch,
        "window_epochs": window,
        "derived_total_epochs": total_epochs,
        "train_mel": {"mean": mean_tm, "ema": ema_tm, "slope": slope_tm, "delta_pct": delta_tm, "verdict": verdict_tm},
        "val_mel":   {"mean": mean_vm, "ema": ema_vm, "slope": slope_vm, "delta_pct": delta_vm, "verdict": verdict_vm},
        "gap":       {"mean": gap_mean, "ema": gap_ema, "slope": gap_slope, "verdict": gap_verdict},
        "gan":       {"gen_std": gen_std, "disc_std": disc_std, "corr": corr_gd, "verdict": gan_verdict},
        "best_epoch": best_epoch,
        "best_val_mel": (best_val if best_epoch is not None else float('nan')),
        "since_best_epochs": since_best_epochs,
        "since_best_time": since_best_time,  # wall-clock if timestamps exist
        "projection": {
            "to_total_epochs": {"epoch": proj_to, "val_mel": proj_end, "r2": r2_end, "delta_from_now": delta_to_end},
            "to_next_horizon": {"epoch": cur_epoch + horizon, "val_mel": proj_next, "r2": r2_next, "delta_from_now": delta_next},
        },
        "status": status,
    }
    return summary, per_epoch

def render_markdown(report_path: Path, log_paths: List[Path], summary: dict, per_epoch: dict):
    p = report_path
    p.parent.mkdir(parents=True, exist_ok=True)
    def fmt(x):
        return "nan" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.6f}"
    with p.open('w', encoding='utf-8') as f:
        w = lambda s="": f.write(s + "\n")
        w(f"# Training Analysis")
        w(f"Logs: " + ", ".join(str(lp) for lp in log_paths))
        w()
        w(f"**Epochs parsed:** {len(per_epoch)}  |  **Current epoch:** {summary['current_epoch']}  |  **Window:** last {summary['window_epochs']} epoch(s)")
        w(f"**Derived total epochs:** {summary.get('derived_total_epochs')}")
        w()
        w(f"**Best epoch so far:** {summary.get('best_epoch')}  (val_mel = {fmt(summary.get('best_val_mel'))})  "
          f"|  **Since best:** {summary.get('since_best_epochs')} epoch(s)"
          + (f" (~{summary.get('since_best_time')})" if summary.get('since_best_time') else ""))
        w()
        vm, tm, gap = summary["val_mel"], summary["train_mel"], summary["gap"]
        w("## Trend (last window)")
        w("| Metric | mean | EMA(0.2) | slope/epoch | Δ% (window) | verdict |")
        w("|---|---:|---:|---:|---:|---|")
        w(f"| train_mel | {fmt(tm['mean'])} | {fmt(tm['ema'])} | {fmt(tm['slope'])} | {fmt(tm['delta_pct'])} | {tm['verdict']} |")
        w(f"| val_mel   | {fmt(vm['mean'])} | {fmt(vm['ema'])} | {fmt(vm['slope'])} | {fmt(vm['delta_pct'])} | {vm['verdict']} |")
        w(f"| gap       | {fmt(gap['mean'])} | {fmt(gap['ema'])} | {fmt(gap['slope'])} |  —  | {gap['verdict']} |")
        w()
        gan = summary["gan"]
        w("## GAN stability")
        w(f"- gen std: {fmt(gan['gen_std'])}, disc std: {fmt(gan['disc_std'])}, corr(gen,disc): {fmt(gan['corr'])} → **{gan['verdict']}**")
        w()
        proj = summary["projection"]
        pe, pn = proj["to_total_epochs"], proj["to_next_horizon"]
        w("## Projection")
        w(f"- To total epochs (e={pe['epoch']}): val_mel ≈ {fmt(pe['val_mel'])} (Δ from now {fmt(pe['delta_from_now'])}), R²={fmt(pe['r2'])}")
        w(f"- Next horizon (e={pn['epoch']}): val_mel ≈ {fmt(pn['val_mel'])} (Δ from now {fmt(pn['delta_from_now'])}), R²={fmt(pn['r2'])}")
        w()
        w(f"## Status: **{summary['status']}**")
        if summary['status'] == "CONVERGING":
            w("- Keep training; reconsider LR schedule only if improvement slows in the next window.")
        elif summary['status'] == "PLATEAU":
            w("- Hold for ~5–10 more epochs; if no new best, decay LR ×0.5 or early-stop.")
        elif summary['status'] == "DIVERGING":
            w("- Consider LR ×0.5 and check data/regularisation; monitor gap and adversarial stability.")
        w()
        w("—")
        w("_Generated by analyze_training.py_")

def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="One or more training log files")
    ap.add_argument("--total_epochs", type=int, default=None, help="Override: total planned epochs (else auto-derived)")
    ap.add_argument("--window", type=int, default=None, help="Epoch window (default: 20% of parsed epochs, min 6)")
    ap.add_argument("--horizon", type=int, default=10, help="Projection horizon in epochs (default 10)")
    ap.add_argument("--out_dir", type=str, default=None, help="Where to write reports (default: alongside first log)")
    ap.add_argument("--plots", action="store_true", help="Write PNG charts if matplotlib is available")
    args = ap.parse_args()

    log_paths = [Path(p) for p in args.logs]
    for p in log_paths:
        if not p.exists():
            raise SystemExit(f"Log not found: {p}")

    # Parse all logs
    all_steps: List[StepRecord] = []
    all_vals: List[ValRecord] = []
    epoch_total_candidates: List[int] = []
    for p in log_paths:
        s, v, phases, ep_totals = parse_log_file(p)
        all_steps.extend(s)
        all_vals.extend(v)
        epoch_total_candidates.extend(ep_totals)

    if not all_steps and not all_vals:
        raise SystemExit("No parsable records found. Check log format or regex.")

    summary, per_epoch = summarize(
        all_steps, all_vals, args.total_epochs, epoch_total_candidates, args.window, args.horizon
    )

    # Output dir
    out_dir = Path(args.out_dir) if args.out_dir else log_paths[0].parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON + Markdown
    json_path = out_dir / "training_analysis.json"
    md_path   = out_dir / "training_analysis.md"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    render_markdown(md_path, log_paths, summary, per_epoch)

    # Optional plot
    if args.plots:
        plt = _try_import_matplotlib()
        if plt is not None:
            es = sorted(per_epoch.keys())
            tm = [per_epoch[e]['train_mel'] for e in es]
            vm = [per_epoch[e]['val_mel'] for e in es]
            plt.figure()
            plt.plot(es, tm, marker='o', label='train_mel')
            plt.plot(es, vm, marker='o', label='val_mel')
            plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / "loss_curves.png")
            plt.close()

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {md_path}")
    if args.plots:
        print(f"Wrote: {out_dir / 'loss_curves.png'}")

if __name__ == "__main__":
    main()
