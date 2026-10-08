# CTC Paxlovid cost-effectiveness analysis

Cost-effectiveness of Paxlovid vs usual care in CanTreatCOVID (first 28 days).

## Files
- `helper.py` – data loading, baseline preparation and demographics helpers
- `cea_constants.py` – unit costs (Paxlovid, outpatient, ER, hospital day)
- `step1_data_preparation.py` – per-participant EQ-5D utilities, QALYs, resource use
- `step2_tables.py` – Tables 1–5
- `step3_cea.py` – MICE imputation, Table 6 (ICER, subgroup/sensitivity), net benefit regression

Run in order; outputs go to `results_cost_pax/`.

## Requirements
Python (pandas, numpy, scipy, statsmodels, scikit-learn, rpy2) and R with the `eq5d` package.
Raw REDCap exports are not included; set `CTC_DATA_DIR` to the folder containing them
