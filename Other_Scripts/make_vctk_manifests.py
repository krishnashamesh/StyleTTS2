#!/usr/bin/env python3
"""
make_vctk_manifests.py

Builds train_list.txt, val_list.txt, OOD_texts.txt and spk2id.json from a VCTK-like tree:
  txt_root/p225/p225_001.txt ...
  wav_root/p225/p225_001_mic2.flac ...

NEW:
- Robust audio lookup via --search-pattern (can be given multiple times).
- Optional renaming/moving to a canonical path (--rename-audio + --rename-pattern).
- Optional transcoding if extension changes (--transcode-to-ext [.wav|.flac], --transcode-sr).
- Dry-run support.

Examples
--------
# 1) Keep FLAC, rename *_mic2.flac -> <stem>.flac (in-place), write manifests:
python make_vctk_manifests.py \
  --txt-root /data/VCTK/txt --wav-root /data/VCTK/wav \
  --out-dir Data --ipa --lang en-gb \
  --search-pattern "{speaker}/{stem}_mic2.flac" \
  --search-pattern "{speaker}/{stem}_mic1.flac" \
  --search-pattern "{speaker}/{stem}.flac" \
  --search-pattern "{speaker}/{stem}.wav" \
  --rename-audio --rename-pattern "{speaker}/{stem}.flac"


# 2) Transcode to WAV at 24k under a clean mirror root:
python make_vctk_manifests.py \
  --txt-root /data/VCTK/txt --wav-root /data/VCTK/wav \
  --out-dir Data --ipa --lang en-gb \
  --search-pattern "{speaker}/{stem}_mic2.flac" \
  --rename-audio --rename-out-root /data/VCTK/wav_norm \
  --rename-pattern "{speaker}/{stem}.wav" \
  --transcode-to-ext .wav --transcode-sr 24000

  USE THIS.
  
  python make_vctk_manifests.py \
  --txt-root /opt/apps/StyleTTS2/Data/VCTK/txt --wav-root /opt/apps/StyleTTS2/Data/VCTK/wav \
  --out-dir /opt/apps/StyleTTS2/Data/VCTK --ipa --lang en-gb \
  --search-pattern "{speaker}/{stem}_mic2.flac" \
  --rename-audio --rename-out-root /opt/apps/StyleTTS2/Data/VCTK/wav_norm \
  --rename-pattern "{speaker}/{stem}.wav" \
  --transcode-to-ext .wav --transcode-sr 24000
  

"""
import argparse, re, json, random, os, shutil
from pathlib import Path, PurePosixPath

def to_posix(p: Path, root: Path) -> str:
    return str(PurePosixPath(os.path.relpath(str(p), str(root))))

def clean_text(s: str) -> str:
    s = s.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c','"').replace('\u201d','"')
    s = s.replace('\r', ' ').replace('\n', ' ').strip()
    return re.sub(r'\s+', ' ', s)

def phonemize_or_text(text: str, use_ipa: bool, lang: str) -> str:
    if not use_ipa:
        return text
    try:
        from phonemizer import phonemize
        return phonemize(text, language=lang, backend='espeak',
                         strip=True, preserve_punctuation=False,
                         with_stress=True, njobs=1)
    except Exception as e:
        print(f"[warn] phonemize failed ({e}); keeping raw text.")
        return text

def parse_spkid(name: str) -> int | None:
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else None

def find_audio(wav_root: Path, speaker: str, stem: str, patterns: list[str]) -> Path | None:
    # 1) Try formatted patterns in order
    for pat in patterns:
        candidate = wav_root / pat.format(speaker=speaker, stem=stem)
        if candidate.exists():
            return candidate
    # 2) Fallback glob in speaker dir
    spkdir = wav_root / speaker
    if spkdir.is_dir():
        for p in sorted(spkdir.glob(f"{stem}*")):
            if p.is_file():
                return p
    return None

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def maybe_transcode(src: Path, dst: Path, to_ext: str | None, sr: int | None):
    """
    If to_ext is None or equals src suffix: just copy/move.
    If to_ext != src suffix: decode and save with torchaudio.
    If sr is provided, resample during transcode.
    """
    if to_ext is None or src.suffix.lower() == to_ext.lower():
        if src.resolve() == dst.resolve():
            return  # nothing to do
        ensure_parent(dst)
        shutil.move(str(src), str(dst))
        return

    # Need to transcode
    import torchaudio, torch
    ensure_parent(dst)
    wav, srate = torchaudio.load(str(src))
    if sr and sr != srate:
        wav = torchaudio.functional.resample(wav, srate, sr)
        srate = sr
    torchaudio.save(str(dst), wav, sample_rate=srate)
    # Remove original after successful save
    src.unlink()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--txt-root', required=True, type=Path)
    ap.add_argument('--wav-root', required=True, type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--ood-frac', type=float, default=0.05)
    ap.add_argument('--train-frac-of-rest', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lang', type=str, default='en-gb')
    ap.add_argument('--ipa', action='store_true', help='Convert text to IPA using phonemizer/espeak-ng')

    # Audio search patterns (can pass multiple; order matters)
    ap.add_argument('--search-pattern', action='append', default=[],
                    help='Relative pattern under wav-root, e.g. "{speaker}/{stem}_mic2.flac"')

    # Renaming / normalisation
    ap.add_argument('--rename-audio', action='store_true',
                    help='If set, rename/move (or transcode) found audio to --rename-pattern')
    ap.add_argument('--rename-out-root', type=Path, default=None,
                    help='Alternate root for renamed audio. Default: in-place under --wav-root')
    ap.add_argument('--rename-pattern', type=str, default="{speaker}/{stem}.flac",
                    help='Canonical relative path for renamed audio (with extension)')
    ap.add_argument('--transcode-to-ext', type=str, default=None,
                    help='If set (e.g., ".wav"), transcode to this extension when renaming')
    ap.add_argument('--transcode-sr', type=int, default=None,
                    help='If set, resample during transcode (e.g., 24000)')
    ap.add_argument('--dry-run', action='store_true')

    args = ap.parse_args()
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Reasonable default search order if none given
    if not args.search_pattern:
        args.search_pattern = [
            "{speaker}/{stem}_mic2.flac",
            "{speaker}/{stem}_mic1.flac",
            "{speaker}/{stem}.flac",
            "{speaker}/{stem}.wav",
        ]
    else:
        # argparse uses dash; normalise attribute name
        args.search_pattern = args.search_pattern  # already list

    items = []  # (abs_wav_path_after_rename, text_or_ipa, spk_id)
    spk2id = {}
    rename_root = args.rename_out_root if args.rename_out_root is not None else args.wav_root

    for spk_dir in sorted(p for p in args.txt_root.iterdir() if p.is_dir()):
        spk_name = spk_dir.name  # "p225"
        spk_id = parse_spkid(spk_name)
        if spk_id is None:
            print(f"[skip] cannot parse numeric speaker id from {spk_name}")
            continue
        spk2id[spk_name] = spk_id

        for txt_path in sorted(spk_dir.glob('*.txt')):
            stem = txt_path.stem  # e.g., p225_001
            raw = clean_text(txt_path.read_text(encoding='utf-8', errors='ignore'))
            if not raw:
                continue
            text = phonemize_or_text(raw, args.ipa, args.lang)

            src_audio = find_audio(args.wav_root, spk_name, stem, args.search_pattern)
            if not src_audio:
                print(f"[warn] missing audio for {txt_path} (searched patterns/glob)")
                continue

            # Determine destination (renamed) path
            dst_rel = args.rename_pattern.format(speaker=spk_name, stem=stem)
            dst_path = (rename_root / dst_rel)

            final_audio = src_audio
            if args.rename_audio:
                if args.dry_run:
                    print(f"[dry] {src_audio} -> {dst_path} "
                          f"{'(transcode ' + args.transcode_to_ext + ')' if args.transcode_to_ext else '(move)'}")
                else:
                    maybe_transcode(src_audio, dst_path, args.transcode_to_ext, args.transcode_sr)
                final_audio = dst_path
            else:
                # no rename: still ensure we emit relative to *current* wav_root
                pass

            # Manifest uses paths relative to the directory that actually contains the audio we will train from
            root_for_rel = rename_root if args.rename_audio else args.wav_root
            rel_posix = to_posix(final_audio, root_for_rel)
            items.append((rel_posix, text, spk_id))

    if not items:
        raise SystemExit("No items collected. Check your roots/patterns.")

    random.shuffle(items)
    N = len(items)
    n_ood = int(round(N * args.ood_frac))
    rest = N - n_ood
    n_train = int(round(rest * args.train_frac_of_rest))
    n_val = rest - n_train

    ood = items[:n_ood]
    train = items[n_ood:n_ood+n_train]
    val = items[n_ood+n_train:]

    (args.out_dir / 'train_list.txt').write_text(
        '\n'.join(f"{p}|{t}|{s}" for p,t,s in train) + '\n', encoding='utf-8')
    (args.out_dir / 'val_list.txt').write_text(
        '\n'.join(f"{p}|{t}|{s}" for p,t,s in val) + '\n', encoding='utf-8')
    (args.out_dir / 'OOD_texts.txt').write_text(
        '\n'.join(f"{p}|{t}|{s}" for p,t,s in ood) + '\n', encoding='utf-8')
    (args.out_dir / 'spk2id.json').write_text(json.dumps(spk2id, indent=2), encoding='utf-8')

    print(f"[done] total={N}  ood={len(ood)}  train={len(train)}  val={len(val)}")
    if args.rename_audio:
        print(f"[info] audio root for manifests: {rename_root}")
    else:
        print(f"[info] audio root for manifests: {args.wav_root}")

if __name__ == '__main__':
    main()
