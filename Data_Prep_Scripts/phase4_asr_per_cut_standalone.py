#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 — ASR per Cut (standalone, no external script)
- Loads a NeMo ASR model (Parakeet v3 / FastConformer / etc.)
- Transcribes each cut independently (no buffering, no dedupe)
- Auto-resamples to model's required sample rate
- Emits per-utt .json/.txt, consolidated index.json, qc.csv, and train_list.txt

Requirements (in your 'nemo' env):
  pip install nemo_toolkit['asr'] soundfile numpy scipy torch

Typical run:
  python phase4_asr_per_cut_standalone.py \
    --cuts_dir /opt/apps/Training/cuts \
    --out_dir  /opt/apps/Training/asr_per_cut \
    --asr_model nvidia/parakeet-tdt-0.6b-v3  \
    --device cuda \
    --jobs 1 \
    --emit_words \
    --train_list_path /opt/apps/Training/manifests/train_list.txt
"""

import argparse, os, re, json, csv, sys, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import torch
    from nemo.collections.asr.models import ASRModel
except Exception as e:
    print("[ERROR] NeMo ASR not available. Install nemo_toolkit['asr'] in this env.", file=sys.stderr)
    raise

try:
    # high-quality resampler
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None  # fallback later if needed


# --------------------------- IO helpers ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_wavs(cuts_dir: Path) -> List[Path]:
    return sorted(cuts_dir.glob("*utt_*.wav"))

def infer_spk_from_name(path: Path) -> str:
    m = re.search(r"_spk[_\-]?(\d+)\.wav$", path.name)
    if m: return f"spk{m.group(1)}"
    m = re.search(r"(spk\d+)", path.stem)
    return m.group(1) if m else "spk0"

def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    return x, sr

def to_model_sr(x: np.ndarray, sr: int, model_sr: int) -> Tuple[np.ndarray, int]:
    if sr == model_sr:
        return x, sr
    if resample_poly is None:
        # simple (but acceptable on short cuts) fallback
        ratio = model_sr / float(sr)
        n = max(1, int(round(len(x) * ratio)))
        xp = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
        x2 = np.interp(np.linspace(0.0, 1.0, num=n, dtype=np.float32), xp, x).astype(np.float32)
        return x2, model_sr
    # HQ resample with polyphase
    g = math.gcd(sr, model_sr)
    up = model_sr // g
    down = sr // g
    x2 = resample_poly(x, up, down).astype(np.float32)
    return x2, model_sr


# --------------------------- Confidence ---------------------------

def frame_rms_activity(x: np.ndarray, sr: int, win_ms=25, hop_ms=10, thr_percentile=20.0) -> float:
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    # stride framing
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win),
        strides=(x.strides[0]*hop, x.strides[0])
    ).astype(np.float32)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    thr = np.percentile(rms, max(1.0, min(99.0, thr_percentile)))
    return float((rms > thr).mean()) if rms.size else 0.0

def words_per_sec(text: str, dur_s: float) -> float:
    if dur_s <= 0: return 0.0
    w = len([t for t in re.split(r"\s+", text.strip()) if t])
    return w / dur_s

def compute_confidence(text: str, x: np.ndarray, sr: int, dur_s: float,
                       target_wps=2.5, short_sec=0.7, short_penalty=0.7,
                       thr_percentile=20.0) -> Tuple[float, Dict[str, Any]]:
    if dur_s <= 0:
        return 0.0, {"reason": "zero_dur"}
    acoustic = frame_rms_activity(x, sr, thr_percentile=thr_percentile)
    wps = words_per_sec(text, dur_s) if text else 0.0
    length_conf = min(1.0, wps / target_wps) if text else 0.0
    conf = 0.5 * acoustic + 0.5 * length_conf
    flags = []
    if not text:
        conf *= 0.5
        flags.append("empty_text")
    if dur_s < short_sec:
        conf *= short_penalty
        flags.append("short_clip")
    return float(round(conf, 3)), {"acoustic": round(acoustic,3), "wps": round(wps,3), "flags": flags}

_PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)
def _normalize_for_empty(s: str) -> str:
    """
    Light normalization for gating: keep apostrophes, drop other punctuation,
    collapse whitespace, lower-case.
    """
    if not s:
        return ""
    s2 = _PUNCT_RE.sub(" ", s).lower()
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _normalize_words(raw) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None
    out = []
    for w in raw:
        # case A: plain string
        if isinstance(w, str):
            out.append({"word": w})
            continue
        # case B: dict-like
        if isinstance(w, dict):
            word = w.get("word") or w.get("text") or str(w)
            start = (w.get("start") or w.get("start_time") or
                     w.get("start_offset") or None)
            end   = (w.get("end") or w.get("end_time") or
                     w.get("end_offset") or None)
            conf  = w.get("conf") or w.get("confidence")
            out.append({"word": word, "start": start, "end": end, "conf": conf})
            continue
        # case C: object with attributes
        word = getattr(w, "word", None) or getattr(w, "text", None) or str(w)
        start = (getattr(w, "start", None) or getattr(w, "start_time", None) or
                 getattr(w, "start_offset", None))
        end   = (getattr(w, "end", None) or getattr(w, "end_time", None) or
                 getattr(w, "end_offset", None))
        conf  = getattr(w, "conf", None) or getattr(w, "confidence", None)
        out.append({"word": word, "start": start, "end": end, "conf": conf})
    return out if out else None

# --------------------------- ASR Wrapper ---------------------------

class NemoASR:
    def __init__(self, model_name_or_path: str, device: str = "cuda", greedy: bool = True):
        self.device = device
        # load
        self.model: ASRModel = ASRModel.from_pretrained(model_name=model_name_or_path)
        self.model.freeze()
        self.model.to(self._torch_device())

        # pick greedy/beam if available; default greedy for speed
        if hasattr(self.model, "change_decoding_strategy"):
            try:
                if greedy:
                    self.model.change_decoding_strategy(decoding_cfg={"strategy": "greedy"})
                else:
                    self.model.change_decoding_strategy(decoding_cfg={"strategy": "beamsearch", "beam_size": 4})
            except Exception:
                pass

        # expected sample rate (most NeMo ASR models: 16 kHz)
        try:
            self.model_sr = int(self.model._cfg.sample_rate)
        except Exception:
            self.model_sr = 16000

        # try to keep apostrophes (depends on model’s normalizer; many preserve by default)
        try:
            tn = getattr(self.model, "decoder", None) or getattr(self.model, "wer", None)
            # no-op here; left for future explicit control if needed
        except Exception:
            pass

    def _torch_device(self):
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def transcribe_one(self, wav_path: Path, emit_words: bool = False) -> Dict[str, Any]:
        """Return {'text': str, 'words': Optional[List[dict]]}"""
        # load + resample
        x, sr = load_audio_mono(wav_path)
        dur = float(len(x)) / float(sr) if sr else 0.0
        x2, sr2 = to_model_sr(x, sr, self.model_sr)

        # NeMo accepts file paths; to guarantee our sr, feed Tensor path via temp WAV if resampled
        # But since we already have original file on disk, leverage NeMo's internal resampling if present.
        # For full control, we do a tensor forward when resampled:
        try_tensor = (sr2 != sr)

        if try_tensor and hasattr(self.model, "transcribe"):
            # direct path call (NeMo may resample internally) — but we already resampled, so use tensor forward if supported
            pass

        # Simplest + robust path: write a small temp WAV only if resampled
        tmp_path: Optional[Path] = None
        audio_path = wav_path
        if sr2 != sr:
            tmp_path = wav_path.parent / f".tmp_{wav_path.stem}_{self.model_sr}hz.wav"
            sf.write(str(tmp_path), x2, self.model_sr)
            audio_path = tmp_path

        import inspect
        def _call_transcribe(return_hyp=False):
            fn = getattr(self.model, "transcribe", None)
            if not callable(fn):
                raise RuntimeError("Model has no callable transcribe()")
            sig = None
            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None
            # choose the right kw for file list
            kw_name = None
            if sig is not None:
                params = sig.parameters
                if "paths2audio_files" in params: kw_name = "paths2audio_files"
                elif "path2audio_files" in params: kw_name = "path2audio_files"
                elif "audio_filepaths" in params: kw_name = "audio_filepaths"
            args = []
            kwargs = {"batch_size": 1}
            if kw_name:
                kwargs[kw_name] = [str(audio_path)]
            else:
                # positional fallback
                args = [[str(audio_path)]]
            # return_hypotheses if supported
            if return_hyp and sig is not None and "return_hypotheses" in sig.parameters:
                # enable timestamps if possible
                if hasattr(self.model, "change_decoding_strategy"):
                    try:
                        self.model.change_decoding_strategy(decoding_cfg={"strategy": "greedy", "compute_timestamps": True, "rnnt_timestamp_type": "all"})
                    except Exception:
                        pass
                kwargs["return_hypotheses"] = True
            return fn(*args, **kwargs)

        try:
            if emit_words:
                hyps = _call_transcribe(return_hyp=True)

                if isinstance(hyps, list) and hyps:
                    if isinstance(hyps[0], str):
                        text = hyps[0]
                        raw_words = None
                    else:
                        text = getattr(hyps[0], "text", "") or ""
                        raw_words = getattr(hyps[0], "words", None)
                else:
                    text = ""
                    raw_words = None

                words = _normalize_words(raw_words)
            else:
                texts = _call_transcribe(return_hyp=False)
                text = texts[0] if isinstance(texts, list) and texts else ""
                words = None
        finally:
            if tmp_path and tmp_path.exists():
                try: tmp_path.unlink()
                except Exception: pass

        return {"text": (text or "").strip(), "words": words, "dur": dur, "sr": sr}


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts_dir", required=True, help="Directory with Phase-3 cuts (utt_*.wav)")
    ap.add_argument("--out_dir", required=True, help="Where to write asr_per_cut/*")
    ap.add_argument("--asr_model", required=True, help="NeMo model name or local .nemo path (e.g., nvidia/parakeet-tdt-0.6b-v3)")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers (I/O bound; keep small for GPU).")
    ap.add_argument("--emit_words", action="store_true", help="Attempt to return word timestamps if the model supports it.")
    ap.add_argument("--target_wps", type=float, default=2.5)
    ap.add_argument("--min_short_sec", type=float, default=0.7)
    ap.add_argument("--short_penalty", type=float, default=0.7)
    ap.add_argument("--rms_thr_percentile", type=float, default=20.0)
    ap.add_argument("--train_list_path", default="", help="Optional path to write train_list.txt")
    ap.add_argument("--min_conf", type=float, default=0.45,
                    help="Drop rows with conf < min_conf when writing train_list.txt")
    ap.add_argument("--min_chars", type=int, default=8,
                    help="Require at least this many non-space characters after light normalization")
    ap.add_argument("--min_words", type=int, default=3,
                    help="Require at least this many whitespace-separated tokens after normalization")
    ap.add_argument("--drop_punct_only", action="store_true", default=True,
                    help="If normalized text is punctuation-only/empty, drop it")

    args = ap.parse_args()

    cuts_dir = Path(args.cuts_dir)
    out_dir  = Path(args.out_dir)
    ensure_dir(out_dir)
    out_json = out_dir / "json"; ensure_dir(out_json)
    out_txt  = out_dir / "txt";  ensure_dir(out_txt)

    wavs = list_wavs(cuts_dir)
    if not wavs:
        print("[ERROR] No utt_*.wav files in cuts_dir.", file=sys.stderr)
        sys.exit(2)

    # Load model once
    asr = NemoASR(args.asr_model, device=args.device, greedy=True)

    # Optional small pool — model is shared; we’ll keep jobs low to avoid VRAM spikes
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = max(1, int(args.jobs))

    results: List[Dict[str, Any]] = []

    def run_one(wav_path: Path) -> Dict[str, Any]:
        utt_id = wav_path.stem
        spk = infer_spk_from_name(wav_path)
        try:
            out = asr.transcribe_one(wav_path, emit_words=args.emit_words)
            text = out["text"]
            dur  = float(out["dur"])
            # Confidence
            x, sr = load_audio_mono(wav_path)
            conf, detail = compute_confidence(
                text, x, sr, dur,
                target_wps=args.target_wps,
                short_sec=args.min_short_sec,
                short_penalty=args.short_penalty,
                thr_percentile=args.rms_thr_percentile
            )

            # Persist per-utt
            (out_txt / f"{utt_id}.txt").write_text(text, encoding="utf-8")
            with open(out_json / f"{utt_id}.json", "w", encoding="utf-8") as jf:
                json.dump({
                    "utt_id": utt_id,
                    "wav": str(wav_path),
                    "text": text,
                    "spk": spk,
                    "dur": round(dur, 3),
                    "conf": conf,
                    "details": detail,
                    "words": out.get("words") if args.emit_words else None
                }, jf, ensure_ascii=False, indent=2)

            return {"utt_id": utt_id, "wav": str(wav_path), "spk": spk,
                    "text": text, "conf": conf, "dur": round(dur,3),
                    "details": detail, "ok": True}

        except Exception as e:
            return {"utt_id": utt_id, "wav": str(wav_path), "spk": spk,
                    "ok": False, "error": str(e)[:600]}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, w) for w in wavs]
        for f in as_completed(futs):
            results.append(f.result())

    results = sorted(results, key=lambda r: r["utt_id"])

    # Consolidated index + QC
    (out_dir / "index.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    qc_path = out_dir / "qc.csv"
    with open(qc_path, "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(["utt_id","spk","dur_s","conf","flags","text_preview","ok"])
        for r in results:
            if r.get("ok"):
                flags = ",".join(r.get("details",{}).get("flags",[]))
                txt = (r.get("text") or "")[:120].replace("\n"," ")
                wr.writerow([r["utt_id"], r["spk"], r.get("dur",0.0), r.get("conf",0.0), flags, txt, "1"])
            else:
                wr.writerow([r["utt_id"], r["spk"], 0.0, 0.0, "error", (r.get("error") or "")[:120], "0"])

    # Optional train_list.txt
    tlp = args.train_list_path.strip()
    if tlp:
        outp = Path(tlp); ensure_dir(outp.parent)
        wrote, dropped = 0, 0
        with open(outp, "w", encoding="utf-8") as f:
            for r in results:
                if not r.get("ok"):
                    continue
                text = (r.get("text") or "").strip()
                conf = float(r.get("conf", 0.0))
                flags = set(r.get("details", {}).get("flags", []))
                # Hard drops
                if "empty_text" in flags:
                    dropped += 1
                    continue
                if conf < args.min_conf:
                    dropped += 1
                    continue
                # Light normalization for emptiness / length checks
                norm = _normalize_for_empty(text)
                char_count = len(norm.replace(" ", ""))
                word_count = len([t for t in norm.split() if t])
                if args.drop_punct_only and char_count == 0:
                    dropped += 1
                    continue
                if char_count < args.min_chars or word_count < args.min_words:
                    dropped += 1
                    continue
                f.write(f"{r['wav']}|{text}|{r['spk']}\n")
                wrote += 1

    # Summary
    oks = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    mean_conf = sum(r.get("conf",0.0) for r in oks) / max(1, len(oks))
    short = sum(1 for r in oks if "short_clip" in r.get("details",{}).get("flags",[]))
    empty = sum(1 for r in oks if "empty_text" in r.get("details",{}).get("flags",[]))
    print(f"[Phase4] processed={len(results)}  ok={len(oks)}  errors={len(errs)}  "
          f"mean_conf={mean_conf:.3f}  short={short}  empty={empty}")
    print(f"[Phase4] wrote: {out_dir/'index.json'}  {qc_path}")
    if tlp:
        print(f"[Phase4] train_list: {tlp}")
        print(f"[Phase4] emitted={wrote}  dropped_by_gates={dropped}  (min_conf={args.min_conf}, "
              f"min_chars={args.min_chars}, min_words={args.min_words}, drop_punct_only={args.drop_punct_only})")

if __name__ == "__main__":
    main()
