#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 — Structure Discovery (buffered) with NeMo MSDD:
- Shard long audio into overlapping "buffers"
- Run NeMo diarization (MSDD) per shard
- Stitch RTTMs back to absolute time
- Drop overlaps (TTS-friendly), merge same-speaker across short gaps
- Enforce min block duration, compute purity
- Emit cleaned RTTM + JSON summary

Outputs:
  {out_dir}/shards/*.wav                    # temp shards (auto-generated)
  {out_dir}/pred_rttms/raw_merged.rttm      # stitched from all shards (pre-clean)
  {out_dir}/pred_rttms/blocks.cleaned.rttm  # overlap-dropped, merged blocks (use this)
  {out_dir}/pred_rttms/blocks.cleaned.json  # machine-readable blocks + purity
"""

import argparse, json, os, subprocess, sys, glob, math, shutil
from dataclasses import dataclass
from typing import List, Tuple

# --- audio I/O (soundfile preferred; fallback to wave) ---
def _read_audio_len_seconds(wav_path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        import wave
        with wave.open(wav_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)

def _write_wav_slice(in_wav: str, out_wav: str, start_s: float, dur_s: float):
    try:
        import soundfile as sf
        data, sr = sf.read(in_wav, dtype="float32", always_2d=False)
        start = int(round(start_s * sr))
        end   = int(round((start_s + dur_s) * sr))
        start = max(0, min(start, len(data)))
        end   = max(0, min(end, len(data)))
        chunk = data[start:end]
        sf.write(out_wav, chunk, sr)
    except Exception:
        # fallback via ffmpeg CLI (robust)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}",
            "-i", in_wav,
            "-c:a", "pcm_s16le", out_wav
        ]
        subprocess.run(cmd, check=True)

# --- RTTM helpers ---
@dataclass
class Seg:
    start: float
    dur: float
    spk: str
    def end(self): return self.start + self.dur

def read_rttm(path: str) -> List[Seg]:
    segs=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            parts = line.strip().split()
            if parts[0] != "SPEAKER": continue
            start=float(parts[3]); dur=float(parts[4]); spk=parts[7]
            segs.append(Seg(start, dur, spk))
    segs.sort(key=lambda s:(s.start, s.start+s.dur))
    return segs

def write_rttm(path: str, rec_name: str, segs: List[Seg]):
    with open(path, 'w', encoding='utf-8') as f:
        for s in segs:
            f.write(f"SPEAKER {rec_name} 1 {s.start:.3f} {s.dur:.3f} <NA> <NA> {s.spk} <NA> <NA>\n")

def _events_from_segs(segs: List[Seg]):
    ev=[]
    for s in segs:
        ev.append((s.start, 'start', s.spk))
        ev.append((s.start+s.dur, 'end', s.spk))
    ev.sort(key=lambda x:(x[0], 0 if x[1]=='end' else 1))
    return ev

def _single_speaker_atoms(segs: List[Seg]) -> List[Seg]:
    """Return minimal segments where exactly one speaker is active."""
    ev=_events_from_segs(segs)
    out=[]; active={}
    last_t=None
    for t,typ,spk in ev:
        if last_t is not None and t>last_t:
            actives=[k for k,v in active.items() if v>0]
            if len(actives)==1:
                out.append(Seg(last_t, t-last_t, actives[0]))
        active[spk]=active.get(spk,0)+ (1 if typ=='start' else -1)
        last_t=t
    # merge exact-touching per speaker
    merged=[]
    for s in out:
        if not merged: merged.append(s); continue
        p=merged[-1]
        if s.spk==p.spk and abs(s.start-(p.start+p.dur))<1e-6:
            merged[-1]=Seg(p.start, p.dur+s.dur, p.spk)
        else:
            merged.append(s)
    return merged

def _overlap_spans(segs: List[Seg]) -> List[tuple]:
    """Intervals where >=2 speakers are active."""
    ev=_events_from_segs(segs)
    spans=[]; active={}
    last_t=None
    for t,typ,spk in ev:
        if last_t is not None and t>last_t:
            if sum(1 for v in active.values() if v>0) >= 2:
                spans.append((last_t, t))
        active[spk]=active.get(spk,0)+ (1 if typ=='start' else -1)
        last_t=t
    return spans

def _union(iv: List[tuple]) -> List[tuple]:
    if not iv: return []
    iv=sorted(iv)
    out=[iv[0]]
    for a,b in iv[1:]:
        A,B=out[-1]
        if a<=B+1e-9:
            out[-1]=(A,max(B,b))
        else:
            out.append((a,b))
    return out

def _subtract(segs: List[Seg], bad: List[tuple]) -> List[Seg]:
    """Subtract 'bad' time windows from single-speaker segs."""
    if not bad: return segs
    out=[]
    for s in segs:
        a=s.start; b=s.start+s.dur
        cur=[(a,b)]
        for x0,x1 in bad:
            nxt=[]
            for u0,u1 in cur:
                if x1<=u0 or x0>=u1:
                    nxt.append((u0,u1))
                else:
                    if u0 < x0: nxt.append((u0,x0))
                    if x1 < u1: nxt.append((x1,u1))
            cur=nxt
        for u0,u1 in cur:
            if u1>u0+1e-6:
                out.append(Seg(u0, u1-u0, s.spk))
    out.sort(key=lambda z:(z.start, z.start+z.dur))
    return out

def merge_same_speaker(segs: List[Seg], max_gap_s: float) -> List[Seg]:
    if not segs: return []
    segs=sorted(segs, key=lambda s:(s.start,s.start+s.dur))
    out=[segs[0]]
    for s in segs[1:]:
        p=out[-1]
        gap = s.start - (p.start+p.dur)
        if s.spk==p.spk and gap<=max_gap_s+1e-6:
            new_end=max(p.start+p.dur, s.start+s.dur)
            out[-1]=Seg(p.start, new_end - p.start, p.spk)
        else:
            out.append(s)
    return out

def filter_min_dur(segs: List[Seg], min_dur: float) -> List[Seg]:
    return [s for s in segs if s.dur>=min_dur-1e-6]

def compute_purity(block: Seg, base_single_spk: List[Seg]) -> float:
    t0, t1 = block.start, block.start+block.dur
    cov=0.0
    for s in base_single_spk:
        if s.spk!=block.spk: continue
        a=max(t0, s.start); b=min(t1, s.start+s.dur)
        if b>a: cov+=(b-a)
    return min(1.0, cov/block.dur) if block.dur>0 else 0.0

# --- Orchestration ---
def build_shards(audio: str, out_dir: str, block_sec: int, hop_sec: int, overlap_sec: float=1.0):
    """Create overlapping shards for long files, plus a manifest for NeMo."""
    os.makedirs(out_dir, exist_ok=True)
    shards_dir=os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    dur=_read_audio_len_seconds(audio)
    if dur <= block_sec:
        # Single shard = whole file (still write one for uniformity)
        shard_meta=[{"path": os.path.join(shards_dir, "shard_000000.wav"),
                     "t0":0.0, "dur":dur}]
        _write_wav_slice(audio, shard_meta[0]["path"], 0.0, dur)
    else:
        shard_meta=[]
        t=0.0
        while t<dur:
            t0=max(0.0, t - overlap_sec/2.0)
            t1=min(dur, t + block_sec + overlap_sec/2.0)
            shard_path=os.path.join(shards_dir, f"shard_{len(shard_meta):06d}.wav")
            _write_wav_slice(audio, shard_path, t0, t1-t0)
            shard_meta.append({"path": shard_path, "t0": t0, "dur": (t1-t0)})
            t += hop_sec
    # Manifest per shard (standard NeMo audio manifest: 1 JSONL per line)
    manifest=os.path.join(out_dir, "manifest_shards.json")
    with open(manifest, "w", encoding="utf-8") as f:
        for m in shard_meta:
            f.write(json.dumps({
                "audio_filepath": m["path"],
                "offset": 0, "duration": None,
                "label": "unknown", "text": ""
            }) + "\n")
    return shard_meta, manifest

def run_nemo_on_manifest(nemo_repo: str, manifest: str, out_dir: str,
                         vad_onset: float, vad_offset: float, vad_pad_onset: float, vad_pad_offset: float,
                         emb_win: float, emb_hop: float, max_speakers: int,
                         msdd_sigmoid: float, device: str):
    infer_py = os.path.join(nemo_repo, "examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_infer.py")
    cfg_path = os.path.join(nemo_repo, "examples/speaker_tasks/diarization/conf/inference")
    cfg_name = "diar_infer_meeting.yaml"

    cmd = [
        sys.executable, infer_py,
        f"--config-path={cfg_path}",
        f"--config-name={cfg_name}",
        f"diarizer.manifest_filepath={manifest}",
        f"diarizer.out_dir={out_dir}",
        f"diarizer.vad.parameters.onset={vad_onset}",
        f"diarizer.vad.parameters.offset={vad_offset}",
        f"diarizer.vad.parameters.pad_onset={vad_pad_onset}",
        f"diarizer.vad.parameters.pad_offset={vad_pad_offset}",
        f"diarizer.speaker_embeddings.parameters.window_length_in_sec={emb_win}",
        f"diarizer.speaker_embeddings.parameters.shift_length_in_sec={emb_hop}",
        f"diarizer.clustering.parameters.max_num_speakers={max_speakers}",
        f"device={device}",
        "diarizer.msdd_model.model_path=diar_msdd_telephonic",
        f"diarizer.msdd_model.parameters.sigmoid_threshold={msdd_sigmoid}",
        # Keep overlaps in raw output; we’ll drop in post
        "diarizer.ignore_overlap=False",
    ]
    print("[NeMo] Running MSDD diarization on shards …")
    subprocess.run(cmd, check=True)

def gather_rttms(out_dir: str) -> List[str]:
    pred = os.path.join(out_dir, "pred_rttms")
    os.makedirs(pred, exist_ok=True)
    return sorted(glob.glob(os.path.join(pred, "*.rttm")))

def stitch_shard_rttm(rttm_files: List[str], shard_meta: List[dict]) -> List[Seg]:
    """Map per-shard RTTMs back to absolute timeline using shard start t0."""
    # Map shard basename -> t0
    idx = { os.path.splitext(os.path.basename(m["path"]))[0]: m["t0"] for m in shard_meta }
    stitched=[]
    for rf in rttm_files:
        base = os.path.splitext(os.path.basename(rf))[0]
        # Some NeMo versions name RTTM using audio stem; handle both exact and loose matches
        # Prefer exact; else try to find shard whose stem is a substring.
        t0 = idx.get(base)
        if t0 is None:
            # loose match
            match = [k for k in idx.keys() if k in base or base in k]
            if match: t0 = idx[match[0]]
            else: t0 = 0.0
        segs = read_rttm(rf)
        for s in segs:
            stitched.append(Seg(start=s.start + t0, dur=s.dur, spk=s.spk))
    stitched.sort(key=lambda s:(s.start, s.start+s.dur))
    return stitched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Phase-0 output: 16 kHz mono WAV")
    ap.add_argument("--out_dir", required=True, help="Output dir, e.g., /opt/apps/Training/nemo_out")
    ap.add_argument("--nemo_repo", required=True, help="Path to NeMo repo, e.g., /opt/apps/NeMo")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])

    # Buffers (shards)
    ap.add_argument("--block-sec", type=int, default=1200, help="Shard length in seconds (e.g., 1200=20 min)")
    ap.add_argument("--block-hop-sec", type=int, default=1200, help="Hop between shard starts (<= block-sec)")
    ap.add_argument("--overlap-sec", type=float, default=1.0, help="Cross-shard overlap to stabilise edges")

    # VAD & embeddings
    ap.add_argument("--vad-onset", type=float, default=0.50)
    ap.add_argument("--vad-offset", type=float, default=0.30)
    ap.add_argument("--vad-pad-onset", type=float, default=0.12)
    ap.add_argument("--vad-pad-offset", type=float, default=0.12)
    ap.add_argument("--emb-win", type=float, default=1.5)
    ap.add_argument("--emb-hop", type=float, default=0.50)

    ap.add_argument("--max-speakers", type=int, default=8)
    ap.add_argument("--msdd-sigmoid", type=float, default=0.60)

    # Post-processing (TTS-friendly)
    ap.add_argument("--merge-silence-ms", type=int, default=700)
    ap.add_argument("--min-block-dur", type=float, default=0.80)
    ap.add_argument("--purity-min", type=float, default=0.90)
    ap.add_argument("--overlap_dilate_ms", type=int, default=80,
                    help="Dilate each MSDD overlap by ±ms before dropping (catches tiny back-channels).")
    ap.add_argument("--short_span_excise_ms", type=int, default=350,
                    help="Excise single-speaker islands shorter than this when flanked by another speaker.")


    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "pred_rttms")
    os.makedirs(pred_dir, exist_ok=True)

    # 1) Build shards + manifest
    shards, manifest = build_shards(
        audio=args.audio,
        out_dir=args.out_dir,
        block_sec=args.block_sec,
        hop_sec=args.block_hop_sec,
        overlap_sec=args.overlap_sec
    )

    # 2) Run NeMo diarization (MSDD) over shards
    run_nemo_on_manifest(
        nemo_repo=args.nemo_repo, manifest=manifest, out_dir=args.out_dir,
        vad_onset=args.vad_onset, vad_offset=args.vad_offset,
        vad_pad_onset=args.vad_pad_onset, vad_pad_offset=args.vad_pad_offset,
        emb_win=args.emb_win, emb_hop=args.emb_hop,
        max_speakers=args.max_speakers,
        msdd_sigmoid=args.msdd_sigmoid, device=args.device
    )

    # 3) Stitch all shard RTTMs to absolute timeline
    rttms = gather_rttms(args.out_dir)
    stitched = stitch_shard_rttm(rttms, shards)
    raw_merged = os.path.join(pred_dir, "raw_merged.rttm")
    rec_name = os.path.splitext(os.path.basename(args.audio))[0]
    write_rttm(raw_merged, rec_name, stitched)

    # 4) Post-process:
    #    (a) atoms where exactly one speaker is active
    single_spk = _single_speaker_atoms(stitched)
    #    (b) dilate-and-drop overlapped regions (to shave back-channels)
    ov = _overlap_spans(stitched)
    if ov and args.overlap_dilate_ms>0:
        d = args.overlap_dilate_ms/1000.0
        ov = _union([(max(0.0,a-d), b+d) for a,b in ov])
        single_spk = _subtract(single_spk, ov)
    #    (c) excise ultra-short islands between different speakers
    if args.short_span_excise_ms>0 and len(single_spk)>=3:
        thr = args.short_span_excise_ms/1000.0
        pruned=[]
        for i,s in enumerate(single_spk):
            if s.dur < thr and 0<i<len(single_spk)-1:
                left,right = single_spk[i-1], single_spk[i+1]
                if left.spk==right.spk!=s.spk:
                    continue  # drop the island
            pruned.append(s)
        single_spk = pruned

    #    (d) merge same-speaker spans and apply min duration
    merged     = merge_same_speaker(single_spk, args.merge_silence_ms / 1000.0)
    cleaned    = filter_min_dur(merged, args.min_block_dur)

    # 5) Compute purity + write cleaned outputs
    blocks=[]
    for b in cleaned:
        pur = compute_purity(b, single_spk)
        blocks.append({
            "start": round(b.start,3),
            "end": round(b.start+b.dur,3),
            "dur": round(b.dur,3),
            "speaker": b.spk,
            "purity": round(pur,3)
        })
    cleaned_rttm = os.path.join(pred_dir, "blocks.cleaned.rttm")
    write_rttm(cleaned_rttm, rec_name, [Seg(b["start"], b["dur"], b["speaker"]) for b in blocks])

    with open(os.path.join(pred_dir, "blocks.cleaned.json"), "w", encoding="utf-8") as jf:
        json.dump({
            "recording": rec_name,
            "audio": args.audio,
            "params": {
                "block_sec": args.block_sec, "block_hop_sec": args.block_hop_sec, "overlap_sec": args.overlap_sec,
                "vad_onset": args.vad_onset, "vad_offset": args.vad_offset,
                "vad_pad_onset": args.vad_pad_onset, "vad_pad_offset": args.vad_pad_offset,
                "emb_win": args.emb_win, "emb_hop": args.emb_hop,
                "max_speakers": args.max_speakers, "msdd_sigmoid": args.msdd_sigmoid,
                "merge_silence_ms": args.merge_silence_ms, "min_block_dur": args.min_block_dur, "purity_min": args.purity_min
            },
            "blocks": blocks
        }, jf, indent=2)

    # 6) Sanity summary
    total = sum(b["dur"] for b in blocks)
    med = sorted(b["dur"] for b in blocks)[len(blocks)//2] if blocks else 0.0
    bad = [b for b in blocks if b["purity"] < args.purity_min]
    print(f"[Phase1:MSDD] blocks={len(blocks)} total_dur={total:.1f}s median_dur={med:.2f}s")
    if bad:
        print(f"[WARN] {len(bad)} blocks below purity_min={args.purity_min}. They'll be excluded in Phase 2.")
    else:
        print("[OK] All blocks meet purity target.")

if __name__ == "__main__":
    main()
