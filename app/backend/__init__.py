"""Application-facing backend for the ECG capstone research demo (stage 14, 27 Oct 2026).

A thin layer over the frozen, stage 10 demo pipeline (demo/infer_partial_ecg.py): it adds a versioned request schema, a
startup asset-integrity check and a local JSON API. It does not reimplement or change any model, feature, calibration or
threshold. Research prototype only - not a medical device.
"""
