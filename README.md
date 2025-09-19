# Assignment-1 — KNN on Bank Marketing (UCI)

## Overview
This repo implements K-Nearest Neighbors on the **Bank Marketing** dataset (UCI). The pipeline:
- downloads dataset from UCI
- preprocesses (one-hot encodes categorical, scales numeric)
- trains and evaluates KNN for multiple k values
- saves confusion matrices, accuracy plot, and a CSV summary

## Files
- `src/knn_bank_marketing.py` — main script (run this)
- `requirements.txt`
- `outputs/` — generated when script runs (confusion matrices, accuracy plot, summary CSV)
- `report.md` — report template to fill for submission

## How to run
1. Create a venv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS / Linux
   venv\Scripts\activate          # Windows
   pip install -r requirements.txt
