#!/bin/bash
# Quick pipeline test before full experiments

echo "=========================================="
echo "Pipeline Sanity Check"
echo "=========================================="
echo ""
echo "Testing:"
echo "  ✓ RefinementDecoder forward_pixels"
echo "  ✓ Focal+Tversky loss"
echo "  ✓ Validation visualization"
echo "  ✓ Checkpointing"
echo ""
echo "Config: 500 sequences, 3 epochs, batch_size=2"
echo "Expected time: ~5-10 minutes"
echo ""
echo "Starting test..."
echo ""

uv run python scripts/train_emberformer.py \
  --config configs/emberformer_test.yaml \
  --gpu 0

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Pipeline test PASSED!"
    echo ""
    echo "Everything works. Safe to run full experiments:"
    echo "  ./run_experiments.sh"
else
    echo "✗ Pipeline test FAILED (exit code: $EXIT_CODE)"
    echo ""
    echo "Check logs and fix issues before running full experiments."
fi
echo "=========================================="

exit $EXIT_CODE
