#!/usr/bin/env python3
import argparse, subprocess, re, sys, json, shlex
from pathlib import Path

SIL_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SIL_END_RE   = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def ffprobe_duration(infile):
    rc, out, err = run([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1", str(infile)
    ])
    if rc != 0:
        print(err.strip(), file=sys.stderr)
        sys.exit(1)
    return float(out.strip())

def detect_silences(infile, noise_db, min_silence):
    """
    Returns list of (start, end, dur) tuples. We fold to mono for detection stability.
    """
    filt = f"pan=mono|c0=0.5*c0+0.5*c1,silencedetect=noise={noise_db}dB:d={min_silence}"
    cmd = ["ffmpeg","-hide_banner","-nostats","-i", str(infile), "-af", filt, "-f","null","-"]
    rc, out, err = run(cmd)

    silences = []
    cur_start = None
    for line in err.splitlines():
        m1 = SIL_START_RE.search(line)
        m2 = SIL_END_RE.search(line)
        if m1:
            cur_start = float(m1.group(1))
        elif m2 and cur_start is not None:
            end = float(m2.group(1)); dur = float(m2.group(2))
            silences.append( (cur_start, end, dur) )
            cur_start = None
    # If stream ends inside a silence, close it at duration end if we can
    return silences

def choose_cut_points(total, silences, min_len, target_len, max_len):
    """
    Build a list of absolute cut boundaries (including 0 and total).
    We cut at the *start* of a silence closest to target_len for each chunk,
    but never earlier than min_len and never later than max_len. If none found,
    we hard-cut at max_len.
    """
    silence_starts = [s for s, e, d in silences]
    cuts = [0.0]
    cur = 0.0
    while True:
        if cur + min_len >= total:
            break
        window_min = cur + min_len
        window_tar = cur + target_len
        window_max = min(cur + max_len, total)

        # candidates inside [window_min, window_max]
        cand = [t for t in silence_starts if window_min <= t <= window_max]
        if cand:
            # pick the one closest to target
            cut = min(cand, key=lambda t: abs(t - window_tar))
        else:
            # no silence in the window → hard cut at window_max
            cut = window_max
        if cut <= cur or cut - cuts[-1] < 1.0:  # sanity: enforce forward progress and >=1s
            cut = min(window_max, total)
        cuts.append(cut)
        cur = cut
        if cur >= total - 1e-3:
            break
    if cuts[-1] < total:
        cuts.append(total)
    # dedup/monotonic
    out = [cuts[0]]
    for t in cuts[1:]:
        if t - out[-1] >= 0.5:
            out.append(t)
    return out

def segment_once(infile, outdir, base, boundaries, codec_copy=True, force_wav=False, samplerate=None, channels=None):
    """
    Use a single ffmpeg call with -f segment and -segment_times (fast).
    boundaries: absolute times [0, t1, t2, ..., total]; we pass t1..t{n-1}.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ext = Path(infile).suffix.lower()
    if force_wav: ext = ".wav"

    outpat = outdir / f"{base}_part_%03d{ext}"
    seg_times = ",".join(f"{t:.3f}" for t in boundaries[1:-1])  # exclude 0 and total

    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y","-i", str(infile), "-map","0:a:0"]
    if codec_copy and not force_wav:
        cmd += ["-c","copy"]
    else:
        # re-encode to PCM WAV (lossless) and enforce channels/sr if asked
        if force_wav and samplerate is None:
            # preserve original SR; ffmpeg will carry through
            pass
        if samplerate: cmd += ["-ar", str(samplerate)]
        if channels:   cmd += ["-ac", str(channels)]
        cmd += ["-c:a","pcm_s16le"]
    cmd += [
        "-f","segment",
        "-segment_times", seg_times,
        "-reset_timestamps","1",
        str(outpat)
    ]
    rc, out, err = run(cmd)
    if rc != 0:
        print("ffmpeg segmentation failed:", err, file=sys.stderr)
        sys.exit(1)
    return outpat

def main():
    ap = argparse.ArgumentParser(description="Split long audio into ~10–12 min chunks, cutting at silences.")
    ap.add_argument("infile", type=Path, help="Input audio (e.g., WAV/MP3), stereo 44.1 kHz")
    ap.add_argument("--out_dir", default="chunks", help="Output directory")
    ap.add_argument("--min_len", type=float, default=600.0, help="Minimum chunk length in seconds (default 600 = 10 min)")
    ap.add_argument("--target_len", type=float, default=660.0, help="Target chunk length in seconds (default 660 = 11 min)")
    ap.add_argument("--max_len", type=float, default=720.0, help="Maximum chunk length in seconds (default 720 = 12 min)")
    ap.add_argument("--noise_db", type=float, default=-30.0, help="Silence threshold in dBFS (default -30)")
    ap.add_argument("--min_silence", type=float, default=0.8, help="Minimum silence duration to consider (seconds)")
    ap.add_argument("--dry_run", action="store_true", help="Only print planned cuts, don’t write files")
    ap.add_argument("--force_wav", action="store_true", help="Re-encode outputs to WAV/PCM (safer across codecs)")
    ap.add_argument("--sr", type=int, default=44100, help="Output sample rate if --force_wav")
    ap.add_argument("--channels", type=int, default=2, help="Output channels if --force_wav")
    args = ap.parse_args()

    infile = args.infile
    base = infile.stem
    total = ffprobe_duration(infile)
    silences = detect_silences(infile, args.noise_db, args.min_silence)
    boundaries = choose_cut_points(total, silences, args.min_len, args.target_len, args.max_len)

    plan = {
        "input": str(infile),
        "duration_sec": round(total, 3),
        "silences_detected": len(silences),
        "cuts_sec": [round(t, 3) for t in boundaries],
        "segments": [
            {"idx": i, "start": round(boundaries[i], 3), "end": round(boundaries[i+1], 3),
             "dur": round(boundaries[i+1]-boundaries[i], 3)}
            for i in range(len(boundaries)-1)
        ]
    }
    print(json.dumps(plan, indent=2))

    if args.dry_run:
        return

    outpat = segment_once(
        infile=infile,
        outdir=args.out_dir,
        base=base,
        boundaries=boundaries,
        codec_copy=(not args.force_wav),
        force_wav=args.force_wav,
        samplerate=args.sr if args.force_wav else None,
        channels=args.channels if args.force_wav else None,
    )
    print(f"Wrote: {outpat.parent}  (pattern: {outpat.name})")

if __name__ == "__main__":
    main()
