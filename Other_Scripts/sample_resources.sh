#!/usr/bin/env bash
set -euo pipefail

# Defaults
INTERVAL="${INTERVAL:-1}"          # seconds between samples
DURATION="${DURATION:-300}"        # total seconds to run
OUTDIR="${OUTDIR:-resource_logs}"  # output folder
TARGET_PID="${TARGET_PID:-}"       # optional PID to track a specific process

mkdir -p "$OUTDIR"

echo "Starting sampling every ${INTERVAL}s for ${DURATION}s -> ${OUTDIR}"
echo "Target PID: ${TARGET_PID:-<none>}"
echo "Started at: $(date -Is)" | tee "$OUTDIR/START.txt"

# ---- CPU (system-wide) ----
# mpstat gives per-CPU + aggregate usage over time
( command -v mpstat >/dev/null 2>&1 || { echo "Install sysstat for mpstat (e.g. sudo apt install sysstat)"; exit 1; } )
mpstat -P ALL "$INTERVAL" $((DURATION/INTERVAL)) > "$OUTDIR/cpu_mpstat.log" &

# ---- Per-process CPU/MEM (optional PID) ----
if [[ -n "${TARGET_PID}" ]]; then
  ( command -v pidstat >/dev/null 2>&1 || { echo "Install sysstat for pidstat (e.g. sudo apt install sysstat)"; exit 1; } )
  # -u CPU, -r memory, -d IO; -h wide; 1-second cadence
  pidstat -h -u -r -d -p "${TARGET_PID}" "$INTERVAL" $((DURATION/INTERVAL)) > "$OUTDIR/pid_${TARGET_PID}_pidstat.log" &
fi

# ---- Top snapshot stream (human-readable) ----
# Good for “like btop/top” scrollback
top -b -d "$INTERVAL" -n $((DURATION/INTERVAL)) > "$OUTDIR/top_stream.log" &

# ---- GPU (system-wide) ----
if command -v nvidia-smi >/dev/null 2>&1; then
  mkdir -p "$OUTDIR"
  INTERVAL="${INTERVAL:-1}"

  # ---- one-time capability snapshots (tolerant) ----
  CAPS_FILE="$OUTDIR/gpu_caps.txt"
  : > "$CAPS_FILE"  # truncate
  for sec in POWER TEMPERATURE CLOCK PERFORMANCE; do
    if nvidia-smi -q -d "$sec" >/dev/null 2>&1; then
      {
        echo "===== $sec ====="
        nvidia-smi -q -d "$sec"
        echo
      } >> "$CAPS_FILE"
    else
      echo "===== $sec (unsupported on this driver) =====" >> "$CAPS_FILE"
    fi
  done
  # Supported clocks (usually available)
  if nvidia-smi -q -d SUPPORTED_CLOCKS >/dev/null 2>&1; then
    {
      echo "===== SUPPORTED_CLOCKS ====="
      nvidia-smi -q -d SUPPORTED_CLOCKS
      echo
    } >> "$CAPS_FILE"
  fi

  # ---- high-level device monitor: power/util/clocks/mem/temp ----
  nvidia-smi dmon -s pucmt -o DT -d "$INTERVAL" > "$OUTDIR/gpu_dmon.log" &
  DMON_PID=$!

  # ---- fine-grained telemetry (CSV) ----
  TEL="$OUTDIR/gpu_telemetry.csv"
  echo "timestamp,index,name,uuid,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,fan.speed,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active" > "$TEL"
  nvidia-smi --query-gpu=timestamp,index,name,uuid,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,fan.speed,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active \
             --format=csv,noheader,nounits \
             -l "$INTERVAL" >> "$TEL" &
  TEL_PID=$!

  # ---- per-process monitor (SM% + MEM%) ----
  nvidia-smi pmon -s um -d "$INTERVAL" -o DT > "$OUTDIR/gpu_pmon.log" &
  PMON_PID=$!
  
fi


# ---- Disk/IO (handy for bottlenecks) ----
if command -v iostat >/dev/null 2>&1; then
  iostat -xz "$INTERVAL" $((DURATION/INTERVAL)) > "$OUTDIR/iostat.log" &
fi

# ---- Memory/pressure snapshots ----
vmstat "$INTERVAL" $((DURATION/INTERVAL)) > "$OUTDIR/vmstat.log" &

# ---- Optional: network throughput ----
if command -v sar >/dev/null 2>&1; then
  sar -n DEV "$INTERVAL" $((DURATION/INTERVAL)) > "$OUTDIR/net_sar.log" &
fi

wait
echo "Finished at: $(date -Is)" | tee "$OUTDIR/DONE.txt"

# Pack it up for sharing
tar -czf "${OUTDIR}.tar.gz" "$OUTDIR"
echo "Bundle ready: ${OUTDIR}.tar.gz"
