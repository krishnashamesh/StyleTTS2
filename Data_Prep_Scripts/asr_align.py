#!/usr/bin/env python3
"""
ASR + word alignment without WhisperX CLI.

- ASR engine: faster_whisper (default) or openai_whisper (optional)
- Alignment: WhisperX aligner (no pyannote).

Outputs in out_dir:
  - segments_raw.json        # whisper segments as produced by the ASR engine
  - segments_aligned.json    # whisperx.align result (segments with word timings)
  - words.jsonl              # one JSON object per word: {"word","start","end"}

Example (import):
    from asr_align import transcribe_and_align
    transcribe_and_align("/opt/apps/bandit/workspace/clip_trim_16k.wav",
                         out_dir="/opt/apps/whisperx/whisperx_out",
                         engine="faster_whisper", model_name="large-v3",
                         compute_type="float16", batch_size=6, beam_size=5, language="en")

Example (run):
    python asr_align.py --audio /opt/apps/bandit/workspace/clip_trim_16k.wav \
                        --out_dir /opt/apps/whisperx/whisperx_out \
                        --engine faster_whisper --model large-v3 \
                        --compute_type float16 --batch_size 6 --beam_size 5 --language en
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ----------------------------
# Utilities
# ----------------------------

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _write_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _write_words_jsonl(segments_aligned: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments_aligned.get("segments", []):
            for w in seg.get("words", []):
                f.write(json.dumps(w, ensure_ascii=False) + "\n")

# ----------------------------
# ASR engines
# ----------------------------

def _asr_faster_whisper(
    audio_path: str | Path,
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 6,     # will be used only if your installed version supports it
    beam_size: int = 5,
    language: Optional[str] = None,
    vad_filter: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Transcribe with faster-whisper on GPU. Returns (segments, language_code).
    Each segment: {"start": float, "end": float, "text": str}
    """
    from faster_whisper import WhisperModel
    import inspect

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    # Build kwargs based on what *this* faster-whisper exposes
    sig = inspect.signature(WhisperModel.transcribe)
    params = sig.parameters

    kwargs = {
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "language": language,       # if None, it will autodetect
    }
    # useful knobs that exist across many versions
    if "chunk_length" in params:
        kwargs["chunk_length"] = 30      # seconds; good default for long files
    if "without_timestamps" in params:
        kwargs["without_timestamps"] = False
    if "temperature" in params:
        kwargs["temperature"] = 0.0      # deterministic; adjust if you like
    if "best_of" in params:
        kwargs["best_of"] = 1            # not used when temperature==0 or with beam search

    # Only pass batch_size if this build supports it
    if "batch_size" in params:
        kwargs["batch_size"] = batch_size

    segments_iter, info = model.transcribe(str(audio_path), **kwargs)
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_iter]
    lang = language or info.language
    return segments, lang


def _asr_openai_whisper(
    audio_path: str | Path,
    model_name: str = "large-v3",   # large-v3 is the best quality available in open-source Whisper
    device: str = "cuda",
    fp16: bool = True,
    beam_size: int = 5,
    language: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Transcribe with openai-whisper (PyTorch). Slower, but some people prefer it.
    Requires: pip install -U openai-whisper
    """
    import torch
    import whisper  # openai-whisper

    model = whisper.load_model(model_name, device=device)
    # For 16 GB VRAM, fp16=True is fine. Use fp16=False if you see OOM.
    result = model.transcribe(
        str(audio_path),
        language=language,
        fp16=fp16 and (device == "cuda" and torch.cuda.is_available()),
        beam_size=beam_size,
        verbose=False,
    )
    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]}
        for s in result.get("segments", [])
    ]
    lang = language or result.get("language", "en")
    return segments, lang

# ----------------------------
# Alignment
# ----------------------------

def _align_with_whisperx(
    segments: List[Dict[str, Any]],
    language: str,
    audio_path: str | Path,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run WhisperX alignment (word-level) on GPU; returns dict with 'segments' incl. words.
    Only uses aligner; does NOT import/execute whisperx.vads/pyannote.
    """
    # Import at call-time to avoid any global side-effects.
    import whisperx

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        segments, align_model, metadata, str(audio_path), device,
        return_char_alignments=False
    )
    return result

# ----------------------------
# Public API
# ----------------------------

def transcribe_and_align(
    audio_path: str | Path,
    out_dir: str | Path,
    engine: str = "faster_whisper",           # "faster_whisper" | "openai_whisper"
    model_name: str = "large-v3",             # try "large-v3" first; fallback to "large-v2" if needed
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 6,
    beam_size: int = 5,
    language: Optional[str] = None,
    vad_filter: bool = False,
) -> Dict[str, Path]:
    """
    Full pipeline: ASR -> save segments_raw.json -> WhisperX align -> save segments_aligned.json & words.jsonl

    Returns dict of output paths.
    """
    audio_path = Path(audio_path)
    out_dir = _ensure_dir(out_dir)

    # 1) ASR
    if engine == "faster_whisper":
        try:
            segments, lang = _asr_faster_whisper(
                audio_path, model_name=model_name, device=device,
                compute_type=compute_type, batch_size=batch_size,
                beam_size=beam_size, language=language, vad_filter=vad_filter
            )
        except Exception as e:
            # If "large-v3" isn’t available in your faster-whisper build, try large-v2 automatically.
            if model_name == "large-v3":
                segments, lang = _asr_faster_whisper(
                    audio_path, model_name="large-v2", device=device,
                    compute_type=compute_type, batch_size=batch_size,
                    beam_size=beam_size, language=language, vad_filter=vad_filter
                )
            else:
                raise
    elif engine == "openai_whisper":
        segments, lang = _asr_openai_whisper(
            audio_path, model_name=model_name, device=device,
            fp16=(compute_type in {"float16", "fp16", "auto"}),
            beam_size=beam_size, language=language
        )
    else:
        raise ValueError(f"Unknown engine: {engine}")

    raw_path = out_dir / "segments_raw.json"
    _write_json({"language": lang, "segments": segments}, raw_path)

    # 2) Alignment
    aligned = _align_with_whisperx(segments, language=lang, audio_path=audio_path, device=device)
    aligned_path = out_dir / "segments_aligned.json"
    _write_json(aligned, aligned_path)

    words_path = out_dir / "words.jsonl"
    _write_words_jsonl(aligned, words_path)

    return {
        "segments_raw": raw_path,
        "segments_aligned": aligned_path,
        "words_jsonl": words_path,
    }

# ----------------------------
# CLI (optional convenience)
# ----------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASR + WhisperX alignment (no WhisperX CLI).")
    p.add_argument("--audio", required=True, help="Path to input audio (your trimmed 16 kHz mono WAV).")
    p.add_argument("--out_dir", required=True, help="Output directory for JSON/JSONL files.")

    p.add_argument("--engine", default="faster_whisper", choices=["faster_whisper", "openai_whisper"])
    p.add_argument("--model", default="large-v3", help="ASR model name (try 'large-v3', fallback to 'large-v2').")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--compute_type", default="float16", help="float16|float32|int8|int8_float16 (faster-whisper only)")
    p.add_argument("--batch_size", type=int, default=6, help="Batch size for faster-whisper.")
    p.add_argument("--beam_size", type=int, default=5, help="Beam size for decoding.")
    p.add_argument("--language", default=None, help="ISO code (e.g., en). If omitted, autodetect.")
    p.add_argument("--vad_filter", action="store_true", help="Enable faster-whisper VAD (usually False if you pre-trimmed).")
    return p

def main():
    ap = _build_argparser()
    args = ap.parse_args()
    out = transcribe_and_align(
        audio_path=args.audio,
        out_dir=args.out_dir,
        engine=args.engine,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        language=args.language,
        vad_filter=args.vad_filter,
    )
    print("Wrote:")
    for k, v in out.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
