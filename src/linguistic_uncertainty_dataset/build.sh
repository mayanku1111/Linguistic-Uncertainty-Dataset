#!/bin/bash
set -euo pipefail

export CURRENT_DIR="$(dirname $0)"
export BUILD_PATH="${CURRENT_DIR}/build"
export CACHE_PATH="${BUILD_PATH}/.cache"
export OUTPUT_PATH="${BUILD_PATH}/output"
export HF_HOME="${CACHE_PATH}/huggingface"
export INPUT_PATH="${CURRENT_DIR}/res"
export PROMPT_PAHT="${CURRENT_DIR}/prompt"

MODE=""
MODEL_ID="Qwen/Qwen3-4B"
INPUT="${INPUT_PATH}/questionlist.csv"
PROMPT="${PROMPT_PAHT}/generate_answer.txt"
CUDA="0"

show_help() {
  cat <<EOF
Usage: $0 [options]

Options:
  -t, --t, -test, --test                          Run test mode
  -ct, --ct, -clean_test, --clean_test            Clean output before test
  -cat, --cat, -clean_all_test, --clean_all_test  Clean cache before test
  -c, --c, -clean, --clean                        Clean all of the cache
  -r, --r, -reset, --reset                        Reset/clean cache
  -m, --m, -model, --model <id>                   Specify model id (e.g. Qwen/Qwen3-0.6B)
  -h, --help                                      Show this help
Examples:
  $0 -test -model Qwen/Qwen3-0.6B
  $0 -c
EOF
}

make_dir() {
    mkdir -p "$BUILD_PATH" "$CACHE_PATH" "$OUTPUT_PATH" "$HF_HOME"
    echo "Build directory creat."
}

test_mode() {
    echo "############### TEST MODE ###############"
    make_dir
    python3 "${CURRENT_DIR}/build.py" -t -model "$MODEL_ID" -input "$INPUT" -prompt "$PROMPT" -cuda "cuda:$CUDA"
}

clean_cache() {
    echo "############### Clean Cache ###############"
    rm -rf "$BUILD_PATH"
    echo "Cache cleaned."
}

clean_cache_without_model() {
    echo "############### Clean Cache without model ###############"
    rm -rf "$OUTPUT_PATH"
    echo "Cache without model cleaned."
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    -c|--c|-clean|--clean)
      MODE="clean"
      shift
      ;;
    -t|--t|-test|--test)
      MODE="test"
      shift
      ;;
    -ct|--ct|-clean_test|--clean_test)
      MODE="clean_test"
      shift
      ;;
    -cat|--cat|-clean_all_test|--clean_all_test)
      MODE="clean_all_test"
      shift
      ;;
    -r|--r|--reset)
      MODE="reset"
      shift
      ;;
    -m|--m|-model|--model)
      if [[ -n "${2:-}" && "${2:0:1}" != "-" ]]; then
        MODEL_ID="$2"
        shift 2
      else
        echo "ERROR: -model requires an argument."
        exit 1
      fi
      ;;
    -input_csv|--input_csv|-input|--input)
      if [[ -n "${2:-}" && "${2:0:1}" != "-" ]]; then
        INPUT="$2"
        shift 2
      else
        echo "ERROR: -input requires an argument."
        exit 1
      fi
      ;;
    -input_prompt|--input_prompt|-prompt|--prompt)
      if [[ -n "${2:-}" && "${2:0:1}" != "-" ]]; then
        PROMPT="$2"
        shift 2
      else
        echo "ERROR: -prompt requires an argument."
        exit 1
      fi
      ;;
    -cuda|--cuda)
      if [[ -n "${2:-}" && "${2:0:1}" != "-" ]]; then
        CUDA="$2"
        shift 2
      else
        echo "ERROR: -prompt requires an argument."
        exit 1
      fi
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

case "$MODE" in
  clean)
    echo "Cleaning build directory..."
    rm -rf "$BUILD_PATH"
    make_dir
    echo "Run"
    # python3 build.py -b -dataset 1.csv
    ;;
  test)
    test_mode
    ;;  
  clean_test)
    clean_cache_without_model
    test_mode
    ;;
  clean_all_test)
    clean_cache
    test_mode
    ;;
  reset)
    clean_cache
    ;;
  *)
    echo "No mode specified."
    show_help
    exit 1
    ;;
esac
