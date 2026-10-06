# ComplaintsPulse — Dataset Documentation

## 1. Dataset Overview

ComplaintsPulse is trained and benchmarked on the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database**, the official US federal repository of consumer complaints against financial institutions.

The processed corpus contains **162,411 validated financial complaints** spanning five major consumer finance product sectors:
- **Credit Reporting**: 91,172 records (56.1%)
- **Debt Collection**: 23,148 records (14.3%)
- **Mortgages & Loans**: 18,990 records (11.7%)
- **Credit Card**: 15,566 records (9.6%)
- **Retail Banking**: 13,535 records (8.3%)

---

## 2. Dataset Characteristics & Schema

* **File Name**: `complaints_processed.csv` (Expected at the project root or linked in `data/`)
* **Columns**:
  * `product`: Cleaned product category label (e.g. `credit_card`, `retail_banking`, `credit_reporting`, `debt_collection`, `mortgages_and_loans`)
  * `narrative`: Pre-scrubbed consumer complaint narrative text
* **Important Note on Timestamps**:
  The processed CFPB text corpus is a static textual dataset without historical transaction filing dates or timestamps. To avoid shipping a 100MB static CSV in GitHub version control, pre-aggregated distributions are stored in `data/analytics_summary.json`, and the raw CSV is excluded via `.gitignore`.

---

## 3. How to Obtain the Raw Dataset

If you wish to re-train the models from scratch or run offline benchmark scripts:
1. Download the CFPB dataset from the official portal at [consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/) or use the preprocessed CSV.
2. Place `complaints_processed.csv` in the root project directory:
   ```bash
   Customer_Complaint_Auto-Tagger/complaints_processed.csv
   ```
3. Run the offline retraining and benchmark scripts:
   ```bash
   python scripts/train_classifier.py
   python scripts/benchmark_similarity.py
   python scripts/benchmark_clustering.py
   python scripts/generate_analytics_data.py
   ```

---

## 4. Precomputed Analytics Artifact

For instant UI rendering (<50ms) without parsing the full 100MB dataset on every startup, precomputed macro distributions are persisted in:
* `data/analytics_summary.json`

This file is tracked in git and powers Page 2 (Complaint Analytics) in the Streamlit dashboard.
