#!/usr/bin/env python3
"""
Emit a clean alignment JSON (and optionally a WhisperX-compatible JSON)
from NFA-aligned word timings (+ optional RTTM + optional seed segments).

Inputs (any combination is fine):
  --nfa_ctm               Path to NFA word CTM (preferred)
  --nfa_json              Path to NFA word JSON: {"words":[{"start":..,"end":..,"word":".."}, ...]}
  --rttm                  Optional diarization RTTM to assign speakers & split on turns
  --seed_segments_jsonl   Optional JSONL with Parakeet segments: {"start":..,"end":..,"text":"..."} per line

Outputs:
  --out_json              Clean alignment JSON (new schema)
  --emit_whisperx         Also emit WhisperX-compatible segments_aligned.json next to out_json

Segmentation knobs:
  --gap_soft       Break if inter-word gap >= this (sec) [default: 1.0]
  --period_gap     After '.', require at least this gap to break [default: 0.8]
  --ellipsis_gap   After '...' or '…', require this gap to break [default: 1.2]
  --min_utt_dur    Min utterance duration to allow standalone [default: 0.35]


python emit_aligned_json.py \
  --nfa_ctm /opt/apps/nfa_out/aligned_words.ctm \
  --rttm /opt/apps/NeMo/nemo_out/pred_rttms/clip_trim_16k.snapped.rttm \
  --audio_relpath clip_trim_24k.wav \
  --gap_soft 1.0 --period_gap 0.8 --ellipsis_gap 1.2 --min_utt_dur 0.35 \
  --out_json /opt/apps/whisperx_compat/aligned_clean.json \
  --emit_whisperx


"""

import json, argparse, sys, re
from pathlib import Path

import os

def load_words_from_ctm(ctm_path):
    """ CTM columns (common): utt channel start dur word [conf]
        We only need start, end, word. Returns list of dicts sorted by start. """
    words=[]
    with open(ctm_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            parts=line.strip().split()
            if len(parts) < 5: continue
            try:
                start=float(parts[2]); dur=float(parts[3]); end=start+dur
            except ValueError:
                continue
            w=parts[4]
            if w == "<eps>":  # common for CTC blanks
                continue
            words.append({"start":start, "end":end, "word":w})
    words.sort(key=lambda x:(x["start"], x["end"]))
    return words

def load_words_from_json(json_path):
    data=json.load(open(json_path, "r", encoding="utf-8"))
    if isinstance(data, dict) and "words" in data:
        words=data["words"]
    elif isinstance(data, list):
        # list of words
        words=data
    else:
        raise ValueError("Unsupported NFA JSON shape; expected dict with 'words' or a list")
    # ensure floats and clean token string
    out=[]
    for w in words:
        try:
            s=float(w["start"]); e=float(w["end"])
            t=str(w.get("word","")).strip()
        except Exception:
            continue
        if t:
            out.append({"start":s, "end":e, "word":t})
    out.sort(key=lambda x:(x["start"], x["end"]))
    return out

def load_rttm(rttm_path):
    """ Parse RTTM -> list of (start, end, speaker) """
    turns=[]
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            if not line.startswith("SPEAKER"): continue
            p=line.strip().split()
            if len(p) < 9: continue
            start=float(p[3]); dur=float(p[4]); end=start+dur; spk=p[7]
            turns.append((start, end, spk))
    turns.sort()
    return turns

def load_seed_segments_jsonl(path):
    """ Each line: {"start":..,"end":..,"text":"..."} """
    segs=[]
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            o=json.loads(ln)
            segs.append({"start": float(o["start"]), "end": float(o["end"]), "text": o.get("text","")})
    segs.sort(key=lambda s:(s["start"], s["end"]))
    return segs

def assign_speaker(word, turns):
    """ Majority overlap speaker for a single word. """
    ws, we = word["start"], word["end"]
    best=None; best_ol=0.0
    for (s,e,spk) in turns:
        if e <= ws or s >= we:  # no overlap
            continue
        ol = min(we,e) - max(ws,s)
        if ol > best_ol:
            best_ol = ol; best = spk
        if e > we and s > we:  # early exit (sorted)
            break
    return best

_PUNCT_HARD = set("?!")
def _is_ellipsis(tok: str) -> bool:
    t = tok.strip()
    return t.endswith("...") or t.endswith("…")

def _join_tokens(tokens):
    # tokens: list[dict{start,end,word}] in time order
    out=[]
    for i, w in enumerate(tokens):
        t = w["word"]
        if not out:
            out.append(t); continue
        # no extra space before punctuation-like tokens
        if t and t[0] in ".,?!;:…":
            out[-1] = out[-1] + t
        else:
            out.append(" " + t)
    return "".join(out).strip()

def segment_words(words, turns=None, seed_segs=None,
                  gap_soft=1.0, period_gap=0.8, ellipsis_gap=1.2, min_dur=0.35):
    """
    Build utterances from a global word list. Split on:
    - speaker changes (if RTTM provided)
    - gaps >= gap_soft
    - '?' or '!' (hard split)
    - '.' with gap >= period_gap
    - '...'/'…' with gap >= ellipsis_gap
    If seed_segs provided: we encourage (not force) splits near seed boundaries.
    """
    utts=[]
    buf=[]; cur_spk=None
    def flush():
        nonlocal buf, cur_spk
        if not buf: return
        s = buf[0]["start"]; e = buf[-1]["end"]
        if e - s >= min_dur or len(buf) == 1:
            utts.append({
                "start": s, "end": e, "speaker": cur_spk,
                "text": _join_tokens(buf),
                "words": buf
            })
        buf=[]

    seed_edges = []
    if seed_segs:
        for ss in seed_segs:
            seed_edges.append(ss["start"])
            seed_edges.append(ss["end"])

    def near_seed_edge(t, tol=0.12):
        if not seed_edges: return False
        # fast check: nearest absolute difference <= tol
        return min(abs(t - se) for se in seed_edges) <= tol

    for i, w in enumerate(words):
        # speaker tag
        spk = assign_speaker(w, turns) if turns else None
        # gap vs previous
        gap = None
        if buf:
            prev = buf[-1]
            gap = w["start"] - prev["end"]
            last_tok = prev["word"].strip()
            is_ell = _is_ellipsis(last_tok)
            is_period = (last_tok.endswith(".") and not is_ell)
            is_hard = (last_tok[-1:] in _PUNCT_HARD)

            boundary = False
            # 1) speaker change
            if turns and spk != cur_spk and cur_spk is not None:
                boundary = True
            # 2) gap-based
            elif gap is not None and gap >= gap_soft:
                boundary = True
            # 3) punctuation-based
            elif is_hard:
                boundary = True
            elif is_ell and gap is not None and gap >= ellipsis_gap:
                boundary = True
            elif is_period and gap is not None and gap >= period_gap:
                boundary = True
            # 4) seed boundary nudging (soft)
            elif seed_segs and (near_seed_edge(w["start"]) or near_seed_edge(prev["end"])):
                # only if there's at least a small pause
                if gap is not None and gap >= 0.06:
                    boundary = True

            if boundary:
                flush()
                cur_spk = spk
        else:
            cur_spk = spk

        # accumulate
        buf.append({"start":w["start"], "end":w["end"], "word":w["word"], "speaker": spk})

    flush()
    # assign sequential ids and normalize speaker labels
    spk_map = {}
    next_spk_id = 0
    for u in utts:
        spk = u["speaker"]
        if spk is None:
            label = None
        else:
            label = spk_map.setdefault(spk, f"speaker_{next_spk_id}")
            if label == f"speaker_{next_spk_id}":
                next_spk_id += 1
        u["speaker"] = label
    # add utt_id
    for i,u in enumerate(utts, start=1):
        u["utt_id"] = f"utt_{i:06d}"
    return utts

def emit_clean_json(utts, audio_relpath, out_path):
    doc = {"audio": audio_relpath, "utterances": []}
    for u in utts:
        doc["utterances"].append({
            "utt_id": u["utt_id"],
            "speaker": u["speaker"],
            "start": round(u["start"], 3),
            "end": round(u["end"], 3),
            "text": u["text"],
            "words": [
                {"start": round(w["start"], 3), "end": round(w["end"], 3), "word": w["word"]}
                for w in u["words"]
            ]
        })
    Path(out_path).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def emit_whisperx_json(utts, out_path):
    segs=[]
    for u in utts:
        segs.append({
            "start": round(u["start"], 3),
            "end": round(u["end"], 3),
            "text": u["text"],
            "words": [
                {"start": round(w["start"], 3), "end": round(w["end"], 3), "word": w["word"]}
                for w in u["words"]
            ]
        })
    doc={"segments": segs}
    Path(out_path).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def main():
    ap=argparse.ArgumentParser()
    # inputs
    ap.add_argument("--nfa_ctm", type=str, default=None, help="NFA-aligned CTM with word timings")
    ap.add_argument("--nfa_json", type=str, default=None, help="NFA-aligned words JSON")
    ap.add_argument("--rttm", type=str, default=None, help="Optional diarization RTTM")
    ap.add_argument("--seed_segments_jsonl", type=str, default=None, help="Optional Parakeet segments JSONL")
    # behavior
    ap.add_argument("--gap_soft", type=float, default=1.0)
    ap.add_argument("--period_gap", type=float, default=0.8)
    ap.add_argument("--ellipsis_gap", type=float, default=1.2)
    ap.add_argument("--min_utt_dur", type=float, default=0.35)
    # outputs
    ap.add_argument("--audio_relpath", type=str, default="clip_trim_24k.wav")
    ap.add_argument("--out_json", type=str, required=True, help="Clean alignment JSON path")
    ap.add_argument("--emit_whisperx", action="store_true", help="Also write segments_aligned.json (WhisperX-compatible)")

    args=ap.parse_args()

    if not args.nfa_ctm and not args.nfa_json:
        print("ERROR: provide either --nfa_ctm or --nfa_json", file=sys.stderr); sys.exit(2)

    if args.nfa_ctm:
        words = load_words_from_ctm(args.nfa_ctm)
    else:
        words = load_words_from_json(args.nfa_json)

    turns = load_rttm(args.rttm) if args.rttm else None
    seeds = load_seed_segments_jsonl(args.seed_segments_jsonl) if args.seed_segments_jsonl else None

    utts = segment_words(
        words, turns=turns, seed_segs=seeds,
        gap_soft=args.gap_soft, period_gap=args.period_gap,
        ellipsis_gap=args.ellipsis_gap, min_dur=args.min_utt_dur
    )

    # emit clean JSON

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    out_clean = emit_clean_json(utts, args.audio_relpath, args.out_json)

    # optional WhisperX-compatible
    if args.emit_whisperx:
        wx_path = str(Path(args.out_json).with_name("segments_aligned.json"))
        emit_whisperx_json(utts, wx_path)

    print(f"Done. Clean JSON: {out_clean}")
    if args.emit_whisperx:
        print(f"WhisperX-compatible: {wx_path}")

if __name__ == "__main__":
    main()
