#!/bin/bash
# 
# MixMatch Training Launcher Script
# Quick launcher for the complete MixMatch training pipeline
#

echo "🎯 MixMatch Training Launcher"
echo "=============================="


# Default parameters
WEBSITES=5
EPOCHS=50
BATCH_SIZE=32
INPUT_DIR_PROXY=""
INPUT_DIR_REQUESTER=""
OUTPUT_DIR=""


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --websites)
            WEBSITES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --input-dir-proxy)
            INPUT_DIR_PROXY="$2"
            shift 2
            ;;
        --input-dir-requester)
            INPUT_DIR_REQUESTER="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --websites N              Number of websites (default: 5)"
            echo "  --epochs N                Training epochs (default: 50)"
            echo "  --batch-size N            Batch size (default: 32)"
            echo "  --input-dir-proxy DIR     Input proxy data directory (required)"
            echo "  --input-dir-requester DIR Input requester data directory (required)"
            echo "  --output-dir DIR          Output directory for all results"
            echo "  --help                    Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


echo "Configuration:"
echo "  Websites: $WEBSITES"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
if [[ -n "$INPUT_DIR_PROXY" ]]; then
    echo "  Input Proxy Dir: $INPUT_DIR_PROXY"
fi
if [[ -n "$INPUT_DIR_REQUESTER" ]]; then
    echo "  Input Requester Dir: $INPUT_DIR_REQUESTER"
fi
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "  Output Dir: $OUTPUT_DIR"
fi
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python3."
    exit 1
fi

# Check required arguments
if [[ -z "$INPUT_DIR_PROXY" ]]; then
    echo "[ERROR] --input-dir-proxy is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ -z "$INPUT_DIR_REQUESTER" ]]; then
    echo "[ERROR] --input-dir-requester is required"
    echo "Use --help for usage information"
    exit 1
fi


# Run the pipeline
echo "[*] Starting MixMatch training pipeline..."
PIPELINE_CMD="python3 scripts/run_pipeline.py --websites $WEBSITES --epochs $EPOCHS --batch-size $BATCH_SIZE --input-dir-proxy $INPUT_DIR_PROXY --input-dir-requester $INPUT_DIR_REQUESTER"
if [[ -n "$OUTPUT_DIR" ]]; then
    PIPELINE_CMD+=" --output-dir $OUTPUT_DIR"
fi
eval $PIPELINE_CMD

# Check the result
if [ $? -eq 0 ]; then
    echo ""
    echo "[OK] Training pipeline completed successfully!"
    echo "  Results are available in the results/ directory"
    echo "  Logs are available in the logs/ directory"
else
    echo ""
    echo "[ERROR] Training pipeline failed!"
    echo "  Check the logs/ directory for error details"
fi
