#!/bin/bash
# Parallel experiments on relela-05 A6000 GPUs

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Starting Parallel EmberFormer Experiments"
echo "=========================================="
echo ""
echo "GPU 0: Focal+Tversky (recommended)"
echo "GPU 1: BCE+Dice (comparison baseline)"
echo ""

# Experiment 1: Focal+Tversky on GPU 0
echo -e "${GREEN}[GPU 0]${NC} Starting Focal+Tversky experiment..."
nohup uv run python scripts/train_emberformer.py \
  --config configs/emberformer.yaml \
  --gpu 0 \
  > logs/focal_tversky_gpu0.log 2>&1 &
PID1=$!
echo -e "${GREEN}[GPU 0]${NC} PID: $PID1"

# Wait a bit to avoid startup conflicts
sleep 5

# Experiment 2: BCE+Dice on GPU 1
echo -e "${BLUE}[GPU 1]${NC} Starting BCE+Dice experiment..."
nohup uv run python scripts/train_emberformer.py \
  --config configs/emberformer_bce_dice.yaml \
  --gpu 1 \
  > logs/bce_dice_gpu1.log 2>&1 &
PID2=$!
echo -e "${BLUE}[GPU 1]${NC} PID: $PID2"

echo ""
echo "=========================================="
echo "Both experiments started!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  tail -f logs/focal_tversky_gpu0.log"
echo "  tail -f logs/bce_dice_gpu1.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Kill both experiments:"
echo "  kill $PID1 $PID2"
echo ""
echo "Process IDs saved to: .experiment_pids"
echo "$PID1" > .experiment_pids
echo "$PID2" >> .experiment_pids
