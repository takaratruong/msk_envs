#!/usr/bin/env bash
set -euo pipefail

# Physical GPU exposed to this process. It becomes cuda:0 inside the process.
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_ENVS="${NUM_ENVS:-1024}"
EXP_PREFIX="${EXP_PREFIX:-stonecourse_allsteps_forward}"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-}"

resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi

checkpoint_args=()
if [[ -n "${WARMSTART_CHECKPOINT}" ]]; then
  checkpoint_args+=(--td3-config.checkpoint-path "${WARMSTART_CHECKPOINT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python -m msk_envs.train.train stonecourse \
  --disable-wandb \
  --exp-prefix "${EXP_PREFIX}" \
  --algo td3 \
  --gpu-id 0 \
  "${resume_args[@]}" \
  --td3-config.num-envs "${NUM_ENVS}" \
  "${checkpoint_args[@]}" \
  env-config:sprinter \
  "$@"
