#!/usr/bin/env bash
# run_pipeline.sh — end-to-end TTS prep pipeline with strict error handling
# Usage:
#   nohup /opt/apps/Training/run_pipeline.sh "Indian Real Estate.mp4" > /opt/apps/Training/nohup.out 2>&1 &
# or:
#   /opt/apps/Training/run_pipeline.sh "Indian Real Estate.mp4"

set -Eeuo pipefail

### --- CONFIG (edit as needed) ---
TRAIN_ROOT="/opt/apps/Training"
SCRIPTS_ROOT="/opt/apps/StyleTTS2/Data_Prep_Scripts"
BANDIT_CKPT="/opt/apps/bandit/workspace/dnr-3s-mus64-l1snr-plus.ckpt"
NEMO_REPO="/opt/apps/NeMo"
ASR_MODEL="nvidia/parakeet-tdt-0.6b-v3"

MAX_SPKS=2

# Phase-2 knobs (you gave these)
MIN_SIL=1.3
SIL_THR_METHOD="percentile"
SIL_THR_VALUE=25
MIN_UTT=1.5
BUCKET_MIN=4.0
BUCKET_TGT=6.0
BUCKET_MAX=8.0
JOIN_GUARD_MS=120
EDGE_GUARD_WIN_MS=250
EDGE_GUARD_SCALE=0.5
HARD_MAX=10.0
MICRO_GAP_MS=100
MICRO_THR_PCTL=35

# Phase-4 gating (you gave these)
PH4_MIN_CONF=0.55
PH4_MIN_CHARS=8
PH4_MIN_WORDS=3

# Phase-5 (you gave these)
P5_MIN_CONF=0.40
P5_LANG="en-us"
### -------------------------------

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"Video Name.ext\"" >&2
  exit 2
fi

VIDEO_NAME="$1"

# Resolve input path
if [[ -f "$VIDEO_NAME" ]]; then
  VIDEO_PATH="$(readlink -f "$VIDEO_NAME")"
else
  # assume it's under TRAIN_ROOT
  VIDEO_PATH="${TRAIN_ROOT}/${VIDEO_NAME}"
fi
if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "[FATAL] Video not found: $VIDEO_PATH" >&2
  exit 3
fi

# Derive run name (spaces -> underscores; strip non-alnum/_)
BASENAME="$(basename "$VIDEO_PATH")"
RUN_NAME="$(printf "%s" "${BASENAME%.*}" | tr ' ' '_' | tr -s '_' | sed 's/[^A-Za-z0-9_]/_/g')"
RUN_DIR="${TRAIN_ROOT}/${RUN_NAME}"
PREFIX="${RUN_NAME}"

# Create folders
mkdir -p "${RUN_DIR}"/{logs,chunks,chunked_output,manifests,proposals,cuts,asr_per_cut,manifests_ipa,nemo_out,mel_cache}
LOG="${RUN_DIR}/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Conda activation (robust)
conda_activate () {
  # shellcheck source=/dev/null
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    . "${HOME}/miniconda3/etc/profile.d/conda.sh"
  else
    echo "[FATAL] conda not found" >&2
    exit 4
  fi
  conda activate "$1"
}

# Error trap
on_err () {
  local exit_code=$?
  echo ""
  echo "[FATAL] Pipeline failed (exit $exit_code). See log: $LOG" >&2
  exit "$exit_code"
}
trap on_err ERR

# Utility to run and log
run () {
  echo -e "\n# --- $* ---" | tee -a "$LOG"
  set -o pipefail
  "$@" 2>&1 | tee -a "$LOG"
}

echo "[INFO] Run dir: $RUN_DIR"
echo "[INFO] Log:     $LOG"

# Common env (memory alloc)
export PROJECT_ROOT="/opt/apps/bandit"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

cd "$TRAIN_ROOT"

### Phase 0 — audio prep + Bandit
AUDIO44="${RUN_DIR}/${PREFIX}_audio_44k_stereo.wav"
run ffmpeg -y -i "$VIDEO_PATH" -map a:0? -vn -sn -dn -ac 2 -ar 44100 -c:a pcm_s16le "$AUDIO44"

conda_activate bandit
run python "${SCRIPTS_ROOT}/smart_split_on_silence.py" "$AUDIO44" --out_dir "${RUN_DIR}/chunks"

# Bandit foreground (so failures stop the pipeline)
run python /opt/apps/bandit/inference.py inference_multiple \
  --ckpt_path="$BANDIT_CKPT" \
  --file_glob="${RUN_DIR}/chunks/*.wav" \
  --model_name=dnr-3s-mus64-l1snr-plus \
  --output_dir="${RUN_DIR}/chunked_output" \
  --include_track_name=True
conda deactivate

# Concat Bandit speech
TRACK_BASE="${PREFIX}_audio_44k_stereo"
LISTFILE="${RUN_DIR}/concat_list.txt"

# collect matches in an array (handles both part_000_0/ and part_000/)
shopt -s nullglob
matches=(
  ${RUN_DIR}/chunked_output/dnr-3s-mus64-l1snr-plus/chunks/${TRACK_BASE}_part_*_*/speech.wav
  ${RUN_DIR}/chunked_output/dnr-3s-mus64-l1snr-plus/chunks/${TRACK_BASE}_part_*/speech.wav
)
shopt -u nullglob

if (( ${#matches[@]} == 0 )); then
  echo "[FATAL] No Bandit speech chunks found. Check this dir:" | tee -a "$LOG"
  echo "  ${RUN_DIR}/chunked_output/dnr-3s-mus64-l1snr-plus/chunks" | tee -a "$LOG"
  exit 5
fi

# sort for deterministic order, then write concat list
printf "%s\n" "${matches[@]}" | sort | while read -r f; do
  printf "file '%s'\n" "$f" >> "$LISTFILE"
done

echo "[INFO] Found ${#matches[@]} speech stems for concat." | tee -a "$LOG"

MERGED="${RUN_DIR}/${PREFIX}_merged_speech.wav"
run ffmpeg -y -f concat -safe 0 -i "$LISTFILE" -c copy "$MERGED"

CLIP16="${RUN_DIR}/${PREFIX}_clip_full_16k.wav"
CLIP24="${RUN_DIR}/${PREFIX}_clip_full_24k.wav"
run ffmpeg -y -i "$MERGED" -ac 1 -ar 16000 "$CLIP16"
run ffmpeg -y -i "$MERGED" -ac 1 -ar 24000 "$CLIP24"

### Phase 1 — structure discovery (MSDD)
conda_activate nemo
run python "${SCRIPTS_ROOT}/phase1_structure_discovery_msdd.py" \
  --audio "$CLIP16" \
  --out_dir "${RUN_DIR}/nemo_out" \
  --nemo_repo "$NEMO_REPO" \
  --device cuda \
  --block-sec 1200 --block-hop-sec 1200 --overlap-sec 1.0 \
  --vad-onset 0.50 --vad-offset 0.30 --vad-pad-onset 0.12 --vad-pad-offset 0.12 \
  --emb-win 1.5 --emb-hop 0.50 \
  --max-speakers "${MAX_SPKS}" \
  --msdd-sigmoid 0.40 \
  --merge-silence-ms 700 \
  --min-block-dur 0.80 \
  --purity-min 1.0 \
  --overlap_dilate_ms 80 \
  --short_span_excise_ms 350
conda deactivate

### Phase 2 — utterance proposals
conda_activate styletts2
run python "${SCRIPTS_ROOT}/phase2_utterance_proposals.py" \
  --audio24k "$CLIP24" \
  --blocks_rttm "${RUN_DIR}/nemo_out/pred_rttms/blocks.cleaned.rttm" \
  --blocks_json "${RUN_DIR}/nemo_out/pred_rttms/blocks.cleaned.json" \
  --out_dir "${RUN_DIR}" \
  --min_silence_sec "${MIN_SIL}" \
  --silence_thr_method "${SIL_THR_METHOD}" --silence_thr_value "${SIL_THR_VALUE}" \
  --min_utt_sec "${MIN_UTT}" \
  --bucket_min_sec "${BUCKET_MIN}" --bucket_target_sec "${BUCKET_TGT}" --bucket_max_sec "${BUCKET_MAX}" \
  --join_guard_ms "${JOIN_GUARD_MS}" \
  --edge_guard_window_ms "${EDGE_GUARD_WIN_MS}" --edge_guard_scale "${EDGE_GUARD_SCALE}" \
  --hard_max_sec "${HARD_MAX}" --micro_gap_ms "${MICRO_GAP_MS}" --micro_thr_percentile "${MICRO_THR_PCTL}"

### Phase 3 — render cuts
run python "${SCRIPTS_ROOT}/phase3_render_cuts.py" \
  --audio24k "$CLIP24" \
  --proposals_json "${RUN_DIR}/proposals/proposals.json" \
  --out_wavs_dir "${RUN_DIR}/cuts" \
  --breadcrumbs "${RUN_DIR}/cutter_breadcrumbs.jsonl" \
  --fade_ms 15 --pad_ms 20
conda deactivate

### Phase 4 — ASR per cut (gated)
conda_activate nemo
run python "${SCRIPTS_ROOT}/phase4_asr_per_cut_standalone.py" \
  --cuts_dir "${RUN_DIR}/cuts" \
  --out_dir  "${RUN_DIR}/asr_per_cut" \
  --asr_model "${ASR_MODEL}" \
  --device cuda --jobs 1 --emit_words \
  --target_wps 3 --min_short_sec 0.7 --short_penalty 0.7 --rms_thr_percentile 20 \
  --train_list_path "${RUN_DIR}/manifests/train_list.txt" \
  --min_conf "${PH4_MIN_CONF}" --min_chars "${PH4_MIN_CHARS}" --min_words "${PH4_MIN_WORDS}" --drop_punct_only
conda deactivate

### Phase 5 — text polish + IPA (with index gating)
conda_activate styletts2
run python "${SCRIPTS_ROOT}/phase5_text_polish_and_ipa.py" \
  --in_manifest "${RUN_DIR}/manifests/train_list.txt" \
  --out_dir "${RUN_DIR}/manifests_ipa" \
  --ipa --lang "${P5_LANG}" \
  --ood-frac 0.05 --train-frac-of-rest 0.95 --seed 42 --normalize \
  --index_json "${RUN_DIR}/asr_per_cut/index.json" --min_conf "${P5_MIN_CONF}"

MEL_CACHE="${RUN_DIR}/mel_cache"

run python "${SCRIPTS_ROOT}/precompute_mels.py" \
  --root "${RUN_DIR}/cuts" \
  --train "${RUN_DIR}/manifests_ipa/train_list.txt" \
  --val "${RUN_DIR}/manifests_ipa/val_list.txt" \
  --ood "${RUN_DIR}/manifests_ipa/OOD_texts.txt" \
  --out "${MEL_CACHE}"

conda deactivate

echo ""
echo "[OK] Pipeline finished successfully."
echo "[OK] Run dir: $RUN_DIR"
echo "[OK] Log:     $LOG"
