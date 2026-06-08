"""H4 (Acquisition‑Order Alignment) analysis pipeline.

Modules:
    predictions_to_eval_results: bridge from `*_predictions.json` (with
        `fast_eval_results` populated) to the per‑checkpoint
        `eval_results.json` format expected by
        `scripts/analyze_acquisition_order.py`.

    run_h4_pipeline: orchestrates the full H4 analysis end‑to‑end (predictions
        → bridge → analyze → CSVs → Figure 6).
"""
