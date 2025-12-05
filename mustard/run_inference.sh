#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
THREADS="10"
WIN_SIZE="100"
STEP="5"
STATIC_PRED_FLAG="0"
MODEL_TYPE="CNN"
CLASS_NUM="2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift 2 ;;
        --genome) GENOME="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --consDir) CONSERVATION="$2"; shift 2 ;;
        --chromList) CHROMOSOMES="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --winSize) WIN_SIZE="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        --staticPredFlag) STATIC_PRED_FLAG="$2"; shift 2 ;;
        --modelType) MODEL_TYPE="$2"; shift 2 ;;
        --classNum) CLASS_NUM="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

case "$MODEL" in
    MuStARD-mirS)  INPUT_MODE="sequence" ;;
    MuStARD-mirF)  INPUT_MODE="RNAfold" ;;
    MuStARD-mirSF) INPUT_MODE="sequence,RNAfold" ;;
    MuStARD-mirSC) INPUT_MODE="sequence,conservation" ;;
    MuStARD-mirFC) INPUT_MODE="RNAfold,conservation" ;;
    *)             INPUT_MODE="sequence,RNAfold,conservation" ;;
esac

ARGS="--winSize $WIN_SIZE --step $STEP --staticPredFlag $STATIC_PRED_FLAG --modelType $MODEL_TYPE --classNum $CLASS_NUM"
ARGS="$ARGS --model $SCRIPT_DIR/mustard_paper/pretrained_models/$MODEL/CNNonRaw.hdf5"
ARGS="$ARGS --inputMode $INPUT_MODE --targetIntervals $INPUT --genome $GENOME"
ARGS="$ARGS --dir $OUTPUT --modelDirName results"
[[ -n "$THREADS" ]] && ARGS="$ARGS --threads $THREADS"
[[ -n "$CONSERVATION" ]] && ARGS="$ARGS --consDir $CONSERVATION"
[[ -n "$CHROMOSOMES" ]] && ARGS="$ARGS --chromList $CHROMOSOMES"

perl "$SCRIPT_DIR/mustard_src/MuStARD.pl" predict $ARGS
