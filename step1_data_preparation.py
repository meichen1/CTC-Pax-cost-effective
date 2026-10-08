"""
STEP 1: Data Preparation for Cost-Effectiveness Analysis of Paxlovid
CanTreatCOVID Study

Steps:
1. Load and merge data using load_and_prepare_data()
2. Combine fpp_ columns into pdd_ columns
3. Extract EQ-5D utility scores at baseline, day 21, day 28 using R eq5d package
4. Calculate QALY over 28 days using trapezoid method
5. Aggregate healthcare utilization variables across time windows
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper import load_and_prepare_data, prepare_baseline_data


# ---------------------------------------------------------------------------
# Helper: combine fpp_ columns into pdd_ columns
# ---------------------------------------------------------------------------

def combine_fpp_into_pdd(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fpp_<suffix> column, if a matching pdd_<suffix> column exists,
    fill missing pdd_ values with fpp_ values (fpp_ rows have NaN pdd_ values
    and vice-versa). Returns df with fpp_ columns dropped.
    """
    fpp_cols = [c for c in df.columns if c.startswith('fpp_')]
    for fpp_col in fpp_cols:
        suffix = fpp_col[len('fpp_'):]          # strip prefix
        pdd_col = 'pdd_' + suffix
        if pdd_col in df.columns:
            # Where pdd_ is NaN, fill from fpp_
            df[pdd_col] = df[pdd_col].combine_first(df[fpp_col])
        else:
            # No matching pdd_ column – rename fpp_ → pdd_
            df[pdd_col] = df[fpp_col]
        df.drop(columns=[fpp_col], inplace=True)
    return df


# ---------------------------------------------------------------------------
# EQ-5D utility score via R eq5d package (Canadian TTO value set)
# ---------------------------------------------------------------------------

def calc_eq5d_utility(mobility, selfcare, activity, pain, anxiety,
                      version='5L', country='Canada', type_='VT'):
    """
    Calculate EQ-5D utility scores for arrays of dimension values using
    the R eq5d package.

    Parameters
    ----------
    mobility, selfcare, activity, pain, anxiety : array-like, int (1-5 for 5L; 1-3 for 3L)
    version  : EQ-5D instrument version ('5L', '3L', or 'Y3L')
    country  : value-set country
    type_    : value-set type ('VT' for 5L; 'TTO' or 'VAS' for 3L)

    Returns
    -------
    np.ndarray of float utility scores (NaN where inputs are missing)
    """
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro

    eq5d_r = importr('eq5d')

    n = len(mobility)
    idx = np.arange(n)

    # Build a mask for rows with all 5 dimensions present
    dims = pd.DataFrame({
        'MO': pd.to_numeric(mobility, errors='coerce'),
        'SC': pd.to_numeric(selfcare, errors='coerce'),
        'UA': pd.to_numeric(activity, errors='coerce'),
        'PD': pd.to_numeric(pain, errors='coerce'),
        'AD': pd.to_numeric(anxiety, errors='coerce'),
    })

    valid = dims.notna().all(axis=1)
    result = np.full(n, np.nan)

    if valid.sum() == 0:
        return result

    dims_valid = dims[valid].astype(int)

    # Build an R data.frame for the eq5d() call
    r_df = ro.DataFrame({
        'MO': ro.IntVector(dims_valid['MO'].tolist()),
        'SC': ro.IntVector(dims_valid['SC'].tolist()),
        'UA': ro.IntVector(dims_valid['UA'].tolist()),
        'PD': ro.IntVector(dims_valid['PD'].tolist()),
        'AD': ro.IntVector(dims_valid['AD'].tolist()),
    })

    scores = eq5d_r.eq5d(
        r_df,
        version=version,
        type=type_,
        country=country,
    )
    scores_arr = np.array(scores, dtype=float)
    result[valid.values] = scores_arr
    return result


# ---------------------------------------------------------------------------
# STEP 1 main function
# ---------------------------------------------------------------------------

def build_step1_dataset() -> pd.DataFrame:
    """
    Build the per-participant STEP 1 dataset containing:
      - EQ-5D dimension columns at baseline, day 21, day 28
      - eq5d_baseline, eq5d_day21, eq5d_day28
      - qaly28
      - Healthcare utilisation variables (day 1-14, day 14-28, day 28-90, day 90-week36)
    """

    # ------------------------------------------------------------------
    # 1. Load data and combine fpp_ → pdd_
    # ------------------------------------------------------------------
    df_raw = load_and_prepare_data()
    df_raw = combine_fpp_into_pdd(df_raw)
    df_baseline = prepare_baseline_data(df_raw)
    ### Remove any columns in baseline starting with 'eq5d_' to avoid confusion with the extracted EQ-5D dimension columns
    eq5d_cols = [c for c in df_baseline.columns if c.startswith('eq5d_')]
    df_baseline.drop(columns=eq5d_cols, inplace=True)

    participant_col = 'participant_id'

    # ------------------------------------------------------------------
    # 2. EQ-5D at BASELINE
    # ------------------------------------------------------------------
    baseline_rows = df_raw[df_raw['redcap_event_name'] == 'Baseline'].copy()

    eq5d_base_dims = ['eq5d_mobility', 'eq5d_scare', 'eq5d_activity', 'eq5d_pain', 'eq5d_anxiety', 'eq5d_vas']
    # Confirm columns exist; use those that are present
    eq5d_base_present = [c for c in eq5d_base_dims if c in baseline_rows.columns]

    baseline_eq5d = baseline_rows[[participant_col] + eq5d_base_present].copy()
    baseline_eq5d = baseline_eq5d.drop_duplicates(subset=[participant_col])

    # Calculate utility score (uses first 5 dimension cols; vas is kept separately)
    base_dim_cols = ['eq5d_mobility', 'eq5d_scare', 'eq5d_activity', 'eq5d_pain', 'eq5d_anxiety']
    base_dim_present = [c for c in base_dim_cols if c in baseline_eq5d.columns]

    if len(base_dim_present) == 5:
        baseline_eq5d['eq5d_utility_baseline'] = calc_eq5d_utility(
            baseline_eq5d['eq5d_mobility'],
            baseline_eq5d['eq5d_scare'],    # SC dimension (self-care)
            baseline_eq5d['eq5d_activity'],
            baseline_eq5d['eq5d_pain'],
            baseline_eq5d['eq5d_anxiety'],
        )
    else:
        baseline_eq5d['eq5d_utility_baseline'] = np.nan

    # ------------------------------------------------------------------
    # 3. EQ-5D at DAY 21
    # ------------------------------------------------------------------
    day21_rows = df_raw[df_raw['redcap_event_name'] == 'Day 21'].copy()

    eq5d_21_dims = ['eq5d_mobility_21', 'eq5d_scare_21', 'eq5d_activity_21',
                    'eq5d_pain_21', 'eq5d_anxiety_21', 'eq5d_vas_21']
    eq5d_21_present = [c for c in eq5d_21_dims if c in day21_rows.columns]

    day21_eq5d = day21_rows[[participant_col] + eq5d_21_present].copy()
    day21_eq5d = day21_eq5d.drop_duplicates(subset=[participant_col])

    dim21_cols = ['eq5d_mobility_21', 'eq5d_scare_21', 'eq5d_activity_21',
                  'eq5d_pain_21', 'eq5d_anxiety_21']
    dim21_present = [c for c in dim21_cols if c in day21_eq5d.columns]

    if len(dim21_present) == 5:
        day21_eq5d['eq5d_utility_21'] = calc_eq5d_utility(
            day21_eq5d['eq5d_mobility_21'],
            day21_eq5d['eq5d_scare_21'],
            day21_eq5d['eq5d_activity_21'],
            day21_eq5d['eq5d_pain_21'],
            day21_eq5d['eq5d_anxiety_21'],
        )
    else:
        day21_eq5d['eq5d_utility_21'] = np.nan

    # ------------------------------------------------------------------
    # 4. EQ-5D at DAY 28  (note: vas column is named eq5d_vas_218 in data)
    # ------------------------------------------------------------------
    day28_rows = df_raw[df_raw['redcap_event_name'] == 'Day 28'].copy()

    eq5d_28_dim_raw = ['eq5d_mobility_28', 'eq5d_scare_28', 'eq5d_activity_28',
                       'eq5d_pain_28', 'eq5d_anxiety_28', 'eq5d_vas_218']
    eq5d_28_present = [c for c in eq5d_28_dim_raw if c in day28_rows.columns]

    day28_eq5d = day28_rows[[participant_col] + eq5d_28_present].copy()
    day28_eq5d = day28_eq5d.drop_duplicates(subset=[participant_col])

    # Rename eq5d_vas_218 → eq5d_vas_28
    if 'eq5d_vas_218' in day28_eq5d.columns:
        day28_eq5d.rename(columns={'eq5d_vas_218': 'eq5d_vas_28'}, inplace=True)

    dim28_cols = ['eq5d_mobility_28', 'eq5d_scare_28', 'eq5d_activity_28',
                  'eq5d_pain_28', 'eq5d_anxiety_28']
    dim28_present = [c for c in dim28_cols if c in day28_eq5d.columns]

    if len(dim28_present) == 5:
        day28_eq5d['eq5d_utility_28'] = calc_eq5d_utility(
            day28_eq5d['eq5d_mobility_28'],
            day28_eq5d['eq5d_scare_28'],
            day28_eq5d['eq5d_activity_28'],
            day28_eq5d['eq5d_pain_28'],
            day28_eq5d['eq5d_anxiety_28'],
        )
    else:
        day28_eq5d['eq5d_utility_28'] = np.nan

    # ------------------------------------------------------------------
    # 4b. EQ-5D at DAY 90
    # ------------------------------------------------------------------
    day90_rows = df_raw[df_raw['redcap_event_name'] == 'Day 90'].copy()

    # Columns at Day 90 share the same base names as Baseline (no numeric suffix)
    eq5d_90_dim_raw = ['eq5d_mobility', 'eq5d_scare', 'eq5d_activity',
                       'eq5d_pain', 'eq5d_anxiety', 'eq5d_vas']
    eq5d_90_present = [c for c in eq5d_90_dim_raw if c in day90_rows.columns]

    day90_eq5d = day90_rows[[participant_col] + eq5d_90_present].copy()
    day90_eq5d = day90_eq5d.drop_duplicates(subset=[participant_col])
    # Rename to _90 suffix to avoid clash with baseline columns
    day90_eq5d.rename(columns={
        'eq5d_mobility': 'eq5d_mobility_90',
        'eq5d_scare':    'eq5d_scare_90',
        'eq5d_activity': 'eq5d_activity_90',
        'eq5d_pain':     'eq5d_pain_90',
        'eq5d_anxiety':  'eq5d_anxiety_90',
        'eq5d_vas':      'eq5d_vas_90',
    }, inplace=True)

    dim90_cols = ['eq5d_mobility_90', 'eq5d_scare_90', 'eq5d_activity_90',
                  'eq5d_pain_90', 'eq5d_anxiety_90']
    dim90_present = [c for c in dim90_cols if c in day90_eq5d.columns]

    if len(dim90_present) == 5:
        day90_eq5d['eq5d_utility_90'] = calc_eq5d_utility(
            day90_eq5d['eq5d_mobility_90'],
            day90_eq5d['eq5d_scare_90'],
            day90_eq5d['eq5d_activity_90'],
            day90_eq5d['eq5d_pain_90'],
            day90_eq5d['eq5d_anxiety_90'],
        )
    else:
        day90_eq5d['eq5d_utility_90'] = np.nan

    # ------------------------------------------------------------------
    # 4c. EQ-5D at WEEK 36
    # ------------------------------------------------------------------
    w36_rows = df_raw[df_raw['redcap_event_name'] == 'Week 36'].copy()

    eq5d_36_present = [c for c in eq5d_90_dim_raw if c in w36_rows.columns]

    w36_eq5d = w36_rows[[participant_col] + eq5d_36_present].copy()
    w36_eq5d = w36_eq5d.drop_duplicates(subset=[participant_col])
    w36_eq5d.rename(columns={
        'eq5d_mobility': 'eq5d_mobility_36',
        'eq5d_scare':    'eq5d_scare_36',
        'eq5d_activity': 'eq5d_activity_36',
        'eq5d_pain':     'eq5d_pain_36',
        'eq5d_anxiety':  'eq5d_anxiety_36',
        'eq5d_vas':      'eq5d_vas_36',
    }, inplace=True)

    dim36_cols = ['eq5d_mobility_36', 'eq5d_scare_36', 'eq5d_activity_36',
                  'eq5d_pain_36', 'eq5d_anxiety_36']
    dim36_present = [c for c in dim36_cols if c in w36_eq5d.columns]

    if len(dim36_present) == 5:
        w36_eq5d['eq5d_utility_36'] = calc_eq5d_utility(
            w36_eq5d['eq5d_mobility_36'],
            w36_eq5d['eq5d_scare_36'],
            w36_eq5d['eq5d_activity_36'],
            w36_eq5d['eq5d_pain_36'],
            w36_eq5d['eq5d_anxiety_36'],
        )
    else:
        w36_eq5d['eq5d_utility_36'] = np.nan

    # ------------------------------------------------------------------
    # 5. Healthcare utilisation – Day 1 to Day 14 (daily diary rows)
    # ------------------------------------------------------------------
    daily_events = [
        'Daily e-Diary(1)', 'Daily e-Diary(2)', 'Daily e-Diary(3)',
        'Daily e-Diary(4)', 'Daily e-Diary(5)', 'Daily e-Diary(6)',
        'Daily e-Diary(7)', 'Daily e-Diary(8)', 'Daily e-Diary(9)',
        'Daily e-Diary(10)', 'Daily e-Diary(11)', 'Daily e-Diary(12)',
        'Daily e-Diary(13)', 'Daily e-Diary(14)',
    ]

    diary_cols = ['pdd_fam_dr', 'pdd_walkin', 'pdd_tel_health', 'pdd_er', 'pdd_hospital']
    diary_present = [c for c in diary_cols if c in df_raw.columns]

    diary_rows = df_raw[df_raw['redcap_event_name'].isin(daily_events)].copy()
    diary_rows[diary_present] = diary_rows[diary_present].apply(pd.to_numeric, errors='coerce').fillna(0)

    day14_hcu = (
        diary_rows.groupby(participant_col)[diary_present]
        .sum()
        .reset_index()
        .rename(columns={
            'pdd_fam_dr':    'day14_fam_dr',
            'pdd_walkin':    'day14_walkin',
            'pdd_tel_health':'day14_tel_health',
            'pdd_er':        'day14_er',
            'pdd_hospital':  'day14_hospital',
        })
    )

    # ------------------------------------------------------------------
    # 6. Healthcare utilisation – Day 14 to Day 28 (day 21 + day 28 rows)
    # ------------------------------------------------------------------
    fup21_cols = ['fup_cov_outpat_visit', 'fup_cov_oupat_visit_numb',
                  'fup_cov_er_visit',     'fup_cov_er_visit_numb',
                  'fup_cov_hosp',         'fup_cov_hosp_numb']
    fup21_present = [c for c in fup21_cols if c in df_raw.columns]

    fup21_rows = df_raw[df_raw['redcap_event_name'] == 'Day 21'][
        [participant_col] + fup21_present
    ].drop_duplicates(subset=[participant_col])

    fup28_cols = ['fup28_cov_outpat_visit', 'fup28_cov_oupat_visit_numb',
                  'fup28_cov_er_visit',     'fup28_cov_er_visit_numb',
                  'fup28_cov_hosp',         'fup28_cov_hosp_numb']
    fup28_present = [c for c in fup28_cols if c in df_raw.columns]

    fup28_rows = df_raw[df_raw['redcap_event_name'] == 'Day 28'][
        [participant_col] + fup28_present
    ].drop_duplicates(subset=[participant_col])

    # ------------------------------------------------------------------
    # 7. Healthcare utilisation – Day 28 to Day 90
    # ------------------------------------------------------------------
    fup90_cols = ['fup90_cov_outpat_visit', 'fup90_cov_oupat_visit_numb',
                  'fup90_cov_er_visit',     'fup90_cov_er_visit_numb',
                  'fup90_cov_hosp',         'fup90_cov_hosp_numb']
    fup90_present = [c for c in fup90_cols if c in df_raw.columns]

    fup90_rows = df_raw[df_raw['redcap_event_name'] == 'Day 90'][
        [participant_col] + fup90_present
    ].drop_duplicates(subset=[participant_col])

    # ------------------------------------------------------------------
    # 8. Healthcare utilisation – Day 90 to Week 36
    # ------------------------------------------------------------------
    fup36_cols = ['fup36_cov_outpat_visit', 'fup36_cov_oupat_visit_numb',
                  'fup36_cov_er_visit',     'fup36_cov_er_visit_numb',
                  'fup36_cov_hosp',         'fup36_cov_hosp_numb']
    fup36_present = [c for c in fup36_cols if c in df_raw.columns]

    fup36_rows = df_raw[df_raw['redcap_event_name'] == 'Week 36'][
        [participant_col] + fup36_present
    ].drop_duplicates(subset=[participant_col])

    # ------------------------------------------------------------------
    # 9. Merge all components into one per-participant dataset
    # ------------------------------------------------------------------
    # Start from unique participant list
    participants = df_raw[[participant_col]].drop_duplicates()

    df_out = participants.copy()
    df_out = df_out.merge(df_baseline[['participant_id','rand_group']], on=participant_col, how='left')
    df_out = df_out.merge(baseline_eq5d, on=participant_col, how='left')
    df_out = df_out.merge(day21_eq5d,   on=participant_col, how='left')
    df_out = df_out.merge(day28_eq5d,   on=participant_col, how='left')
    df_out = df_out.merge(day90_eq5d,   on=participant_col, how='left')
    df_out = df_out.merge(w36_eq5d,     on=participant_col, how='left')
    df_out = df_out.merge(day14_hcu,    on=participant_col, how='left')
    df_out = df_out.merge(fup21_rows,   on=participant_col, how='left')
    df_out = df_out.merge(fup28_rows,   on=participant_col, how='left')
    df_out = df_out.merge(fup90_rows,   on=participant_col, how='left')
    df_out = df_out.merge(fup36_rows,   on=participant_col, how='left')

    # ------------------------------------------------------------------
    # 10. Calculate QALY over 28 days (trapezoid)
    # QALY = ((eq5d_utility_baseline + eq5d_utility_21)*21 + (eq5d_utility_21 + eq5d_utility_28)*7) / (2*365)
    # ------------------------------------------------------------------
    df_out['qaly28'] = (
        (df_out['eq5d_utility_baseline'] + df_out['eq5d_utility_21']) * 21
        + (df_out['eq5d_utility_21'] + df_out['eq5d_utility_28']) * 7
    ) / (2 * 365)

    # ------------------------------------------------------------------
    # 10b. Calculate QALY over 36 weeks (trapezoid, all time points)
    # Time points: day 0 (baseline), day 21, day 28, day 90, day 252 (week 36)
    # Intervals: 21, 7, 62, 162 days
    # QALY_w36 = ((u0+u21)*21 + (u21+u28)*7 + (u28+u90)*62 + (u90+u36)*162) / (2*365)
    # ------------------------------------------------------------------
    df_out['qaly_w36'] = (
        (df_out['eq5d_utility_baseline'] + df_out['eq5d_utility_21']) * 21
        + (df_out['eq5d_utility_21'] + df_out['eq5d_utility_28']) * 7
        + (df_out['eq5d_utility_28'] + df_out['eq5d_utility_90']) * 62
        + (df_out['eq5d_utility_90'] + df_out['eq5d_utility_36']) * 162
    ) / (2 * 365)

    # ------------------------------------------------------------------
    # 11. Round numeric results to 3 decimal places
    # ------------------------------------------------------------------
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[numeric_cols] = df_out[numeric_cols].round(3)

    return df_out, df_baseline


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    
    print("Running STEP 1: Data Preparation...")
    df_step1, df_baseline = build_step1_dataset()

    out_path = os.path.join(os.path.dirname(__file__), 'results_cost_pax', 'cost_pax_step1_dataset.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_step1.to_csv(out_path, index=False)
    df_baseline.to_csv(out_path.replace('cost_pax_step1_dataset.csv', 'cost_pax_baseline_dataset.csv'), index=False)

    print(f"\nDone. Shape: {df_step1.shape}")
    print(f"Saved to: {out_path}")
    print("\nColumn list:")
    for c in df_step1.columns:
        print(f"  {c}")
    print("\nFirst 3 rows (key columns):")
    key_cols = [c for c in ['participant_id', 'eq5d_utility_baseline', 'eq5d_utility_21', 'eq5d_utility_28',
                             'eq5d_utility_90', 'eq5d_utility_36', 'qaly28', 'qaly_w36',
                             'day14_fam_dr', 'day14_walkin', 'day14_tel_health', 'day14_er', 'day14_hospital']
                if c in df_step1.columns]
    print(df_step1[key_cols].head(3).to_string())
