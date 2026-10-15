#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SGLANG_PYTHON:-/home/fluentllmenv/bin/python}"
MODEL_PATH="${SGLANG_MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SGLANG_SERVED_MODEL_NAME:-qwen3-0.6b-local}"
PORT="${SGLANG_SERVER_PORT:-8443}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
READY_TIMEOUT_SECONDS="${SGLANG_READY_TIMEOUT_SECONDS:-1800}"
RUNTIME_DIR="${SGLANG_RUNTIME_DIR:-/tmp/sglang-qwen3-06b}"
LOG_DIR="${SGLANG_LOG_DIR:-/home/$(id -un)/logs}"
LOG_FILE="$LOG_DIR/sglang-qwen3-06b-$PORT.log"
HF_HOME="${HF_HOME:-$RUNTIME_DIR/huggingface}"
PYDEPS_DIR="${SGLANG_PYDEPS_DIR:-$HF_HOME/pydeps-lean-transformers-5.12.1-xgrammar-0.2.1-v2}"
PYDEPS_MARKER="$PYDEPS_DIR/.sglang-hope-deps-ready-v5"
PYPI_INDEX_URL="${SGLANG_PYPI_INDEX_URL:-https://pypi.sankuai.com/simple}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

if [[ ! -r "$REPO_ROOT/python/sglang/__init__.py" ]]; then
  echo "Packaged local SGLang source is missing: $REPO_ROOT/python/sglang/__init__.py" >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$HF_HOME" "$PYDEPS_DIR"

if [[ ! -f "$PYDEPS_MARKER" ]]; then
  echo "Installing current-source Python dependency overlay into $PYDEPS_DIR"
  if command -v uv >/dev/null 2>&1; then
    uv pip install \
      --python "$PYTHON" \
      --target "$PYDEPS_DIR" \
      --index-url "$PYPI_INDEX_URL" \
      --upgrade \
      --no-deps \
      'transformers==5.12.1' \
      'xgrammar==0.2.1' \
      'huggingface-hub>=1.5.0,<2.0' \
      'tokenizers>=0.22.0,<0.23.0' \
      'regex>=2025.10.22' \
      'safetensors>=0.4.3' \
      'sentencepiece>=0.1.91' \
      'gguf==0.19.0' \
      'openai==2.6.1' \
      'soundfile==0.13.1' \
      'sgl-kernel==0.3.17' \
      typer \
      'apache-tvm-ffi>=0.1.9'
  else
    "$PYTHON" -m pip install \
      --target "$PYDEPS_DIR" \
      --index-url "$PYPI_INDEX_URL" \
      --upgrade \
      --no-deps \
      'transformers==5.12.1' \
      'xgrammar==0.2.1' \
      'huggingface-hub>=1.5.0,<2.0' \
      'tokenizers>=0.22.0,<0.23.0' \
      'regex>=2025.10.22' \
      'safetensors>=0.4.3' \
      'sentencepiece>=0.1.91' \
      'gguf==0.19.0' \
      'openai==2.6.1' \
      'soundfile==0.13.1' \
      'sgl-kernel==0.3.17' \
      typer \
      'apache-tvm-ffi>=0.1.9'
  fi
  touch "$PYDEPS_MARKER"
fi

export PYTHONPATH="$PYDEPS_DIR:$REPO_ROOT/hope_compat:$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$RUNTIME_DIR/xdg"
export TRITON_CACHE_DIR="$RUNTIME_DIR/triton"
export CUDA_CACHE_PATH="$RUNTIME_DIR/cuda"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_DIR/torchinductor"
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
export no_proxy="$NO_PROXY"

echo "repo_root=$REPO_ROOT"
echo "python=$PYTHON"
echo "model_path=$MODEL_PATH"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "port=$PORT tp_size=$TP_SIZE"
echo "log_file=$LOG_FILE"
echo "pydeps_dir=$PYDEPS_DIR"
echo "pypi_index_url=$PYPI_INDEX_URL"

"$PYTHON" - <<'PY'
import hashlib
import importlib.metadata
from pathlib import Path

import torch
import transformers
import sglang
from sgl_kernel.utils import is_arch_support_pdl
from transformers import configuration_utils

source = Path(sglang.__file__).resolve()
print(f"sglang_origin={source}")
print(f"sglang_source_sha256={hashlib.sha256(source.read_bytes()).hexdigest()}")
print(f"python_version={__import__('sys').version.split()[0]}")
print(f"torch_version={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"pdl_supported_after_compat={is_arch_support_pdl()}")
print(f"transformers_version={transformers.__version__}")
print(
    "transformers_PreTrainedConfig_available="
    f"{hasattr(configuration_utils, 'PreTrainedConfig')}"
)
for package in ("flashinfer-python", "sgl-kernel", "triton"):
    try:
        print(f"{package}_version={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}_version=NOT_INSTALLED")
PY

"$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --download-dir "$HF_HOME/hub" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --dtype bfloat16 \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests 8 \
  --chunked-prefill-size 2048 \
  --attention-backend triton \
  --grammar-backend none \
  --disable-cuda-graph \
  --trust-remote-code \
  --random-seed 42 \
  --log-level info \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Keep the platform stdout useful while preserving the on-disk traceback. This
# tail process is a sibling of the server, so SGLang's child cleanup will not
# kill it before the final exception is flushed.
tail --pid="$SERVER_PID" -n +1 -F "$LOG_FILE" &
LOG_TAIL_PID=$!

cleanup() {
  if kill -0 "$LOG_TAIL_PID" 2>/dev/null; then
    kill "$LOG_TAIL_PID" 2>/dev/null || true
    wait "$LOG_TAIL_PID" 2>/dev/null || true
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready_deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
until curl --noproxy '*' --fail --silent --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
  server_state="$(ps -o stat= -p "$SERVER_PID" 2>/dev/null | tr -d '[:space:]')"
  if ! kill -0 "$SERVER_PID" 2>/dev/null || [[ -z "$server_state" || "$server_state" == Z* ]]; then
    echo "SGLang server exited before readiness" >&2
    set +e
    wait "$SERVER_PID"
    server_rc=$?
    set -e
    sleep 5
    echo "SGLang server exit code: $server_rc" >&2
    echo "Last SGLang log lines:" >&2
    tail -n 300 "$LOG_FILE" >&2 || true
    exit "$server_rc"
  fi
  if (( SECONDS >= ready_deadline )); then
    echo "SGLang readiness timed out after ${READY_TIMEOUT_SECONDS}s" >&2
    tail -n 200 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 10
done

echo "SGLang service is ready: http://127.0.0.1:$PORT/health"
curl --noproxy '*' --fail --silent --show-error --max-time 30 \
  "http://127.0.0.1:$PORT/v1/models"
echo
curl --noproxy '*' --fail --silent --show-error --max-time 120 \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: SGLANG_OK\"}],\"temperature\":0,\"max_tokens\":16}" \
  "http://127.0.0.1:$PORT/v1/chat/completions"
echo
echo "SGLANG_HOPE_SMOKE_RESULT=PASS"

wait "$SERVER_PID"
