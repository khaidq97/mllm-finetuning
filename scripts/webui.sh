#!/bin/bash
# Script to run LLaMA-Factory Web UI
# This allows you to monitor and manage training through a web interface

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=========================================="
echo "Starting LLaMA-Factory Web UI"
echo "=========================================="
echo ""
echo "The web UI will be available at:"
echo "  - Local: http://localhost:7860"
echo "  - Network: http://0.0.0.0:7860"
echo ""
echo "You can:"
echo "  - Configure training parameters"
echo "  - Start/stop training"
echo "  - Monitor training progress"
echo "  - View logs and metrics"
echo ""
echo "Note: If you see compatibility errors, run:"
echo "  ./scripts/fix_webui_compatibility.sh"
echo ""
echo "Press Ctrl+C to stop the web UI"
echo "=========================================="
echo ""

# Run LLaMA-Factory web UI
llamafactory-cli webui

