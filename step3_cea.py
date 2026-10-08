"""
STEP 3: Cost-Effectiveness Analysis of Paxlovid - CanTreatCOVID Study

1. Import step-1 data and demographic covariates (extract_demographics).
2. Compute per-participant total cost (day1-day28) and QALY over 28 days.
3. Impute missing values with MICE (sklearn IterativeImputer).
4. Table 6: cost-effectiveness results with ICER and probability cost-effective
   at WTP = $30 000, $50 000, $70 000.
   Sub-analyses: complete cases, age>=75, age>=65, multimorbidity>=2, multimorbidity>=3.
5. Net-benefit regression:
       NB = qaly28 * WTP - total_cost_28d
   Adjusted for: age, sex (female vs male, excluding intersex),
   race (non-white vs white), income binary (<=2 vs >2).
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from step1_data_preparation import build_step1_dataset
from helper import extract_demographics, extract_randomization_data
from cea_constants import COST_PAX, COST_OUTPAT, COST_ER, COST_HOSPITAL

RESULTS_DIR = os.path.join(HERE, 'results_cost_pax')

GROUPS = ['Paxlovid', 'Usual Care']

# ── bootstrap settings ──────────────────────────────────────────────────────
N_BOOT = 1000
SEED   = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mean_se(arr, digits=3):
    arr = pd.to_numeric(arr, errors='coerce').dropna().values
    if len(arr) == 0:
        return 'N/A'
    m  = arr.mean()
    se = arr.std() / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return f"{m:.{digits}f} ({se:.{digits}f})"


def _boot_ci_diff(a, b, n_boot=N_BOOT, seed=SEED, digits=3):
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    a = np.array(pd.to_numeric(a, errors='coerce').dropna(), dtype=float)
    b = np.array(pd.to_numeric(b, errors='coerce').dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    rng   = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a), replace=True).mean()
             - rng.choice(b, len(b), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return round(float(lo), digits), round(float(hi), digits)


def _ci_str(lo, hi, digits=3):
    if isinstance(lo, float) and np.isnan(lo):
        return 'N/A'
    return f"({lo:.{digits}f}, {hi:.{digits}f})"


# ---------------------------------------------------------------------------
# 1. Compute per-participant total cost (day1 - day28)
# ---------------------------------------------------------------------------

def compute_total_cost_28d(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series (index = df.index) of total healthcare cost per participant
    over day 1 – day 28, plus Paxlovid drug cost for Paxlovid arm.

    Cost components:
      day1-14 : day14_fam_dr, day14_walkin (×OUTPAT), day14_er (×ER),
                day14_hospital (×HOSPITAL)
      14-21   : fup_cov_oupat_visit_numb (×OUTPAT), fup_cov_er_visit_numb (×ER),
                fup_cov_hosp_numb (×HOSPITAL);  zero-fill by binary col
      21-28   : fup28_cov_oupat_visit_numb, fup28_cov_er_visit_numb, fup28_cov_hosp_numb
    """

    def _num(col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').fillna(0)
        return pd.Series(0.0, index=df.index)

    # zero-fill _numb where binary==0  (not-visited → 0 visits)
    def _fill_numb(bin_col, numb_col):
        if bin_col not in df.columns or numb_col not in df.columns:
            return pd.Series(0.0, index=df.index)
        filled = pd.to_numeric(df[numb_col], errors='coerce').copy()
        no_visit = (pd.to_numeric(df[bin_col], errors='coerce') == 0)
        filled.loc[no_visit] = 0.0
        return filled.fillna(0)

    cost = pd.Series(0.0, index=df.index)

    # day1-14
    cost += (_num('day14_fam_dr') + _num('day14_walkin')) * COST_OUTPAT
    cost += _num('day14_er')       * COST_ER
    cost += _num('day14_hospital') * COST_HOSPITAL

    # 14-21
    cost += _fill_numb('fup_cov_outpat_visit',  'fup_cov_oupat_visit_numb') * COST_OUTPAT
    cost += _fill_numb('fup_cov_er_visit',       'fup_cov_er_visit_numb')   * COST_ER
    cost += _fill_numb('fup_cov_hosp',           'fup_cov_hosp_numb')       * COST_HOSPITAL

    # 21-28
    cost += _fill_numb('fup28_cov_outpat_visit', 'fup28_cov_oupat_visit_numb') * COST_OUTPAT
    cost += _fill_numb('fup28_cov_er_visit',      'fup28_cov_er_visit_numb')   * COST_ER
    cost += _fill_numb('fup28_cov_hosp',          'fup28_cov_hosp_numb')       * COST_HOSPITAL

    # Paxlovid drug cost
    pax_mask = df['rand_group'] == 'Paxlovid'
    cost.loc[pax_mask] += COST_PAX

    return cost


# ---------------------------------------------------------------------------
# 2. Get per-participant QALY over 28 days
# ---------------------------------------------------------------------------

def get_qaly_28d(df: pd.DataFrame) -> pd.Series:
    """Return qaly28 column as a numeric Series."""
    return pd.to_numeric(df['qaly28'], errors='coerce')


# ---------------------------------------------------------------------------
# 3. MICE imputation
# ---------------------------------------------------------------------------

def mice_impute(df: pd.DataFrame, numeric_cols: list, n_datasets: int = 5,
                iterations: int = 5, seed: int = SEED) -> pd.DataFrame:
    """
    Impute numeric_cols in df using sklearn IterativeImputer (MICE equivalent).
    Returns a single imputed DataFrame (averaged over n_datasets imputations).
    """
    from sklearn.experimental import enable_iterative_imputer   # noqa
    from sklearn.impute import IterativeImputer

    df_in = df[numeric_cols].copy().reset_index(drop=True)
    for c in numeric_cols:
        df_in[c] = pd.to_numeric(df_in[c], errors='coerce')

    # Drop columns that are entirely NaN
    all_nan_cols = [c for c in df_in.columns if df_in[c].isna().all()]
    if all_nan_cols:
        print(f"  Dropping all-NaN columns from MICE: {all_nan_cols}")
        df_in = df_in.drop(columns=all_nan_cols)
        numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]

    # Run n_datasets imputations and average
    imp_results = []
    for i in range(n_datasets):
        imputer = IterativeImputer(max_iter=iterations, random_state=seed + i,
                                   skip_complete=True)
        arr = imputer.fit_transform(df_in.values)
        imp_results.append(arr)

    avg = np.mean(np.stack(imp_results, axis=0), axis=0)
    df_out = df.copy()
    for j, c in enumerate(numeric_cols):
        df_out[c] = avg[:, j]
    return df_out


# ---------------------------------------------------------------------------
# 4. Build analysis dataset (step1 + demographics + cost + qaly)
# ---------------------------------------------------------------------------

def build_analysis_dataset(df_step1: pd.DataFrame,
                            df_raw_for_demo: pd.DataFrame) -> pd.DataFrame:
    """
    Merge step1 data with demographics, compute total_cost_28d and qaly28,
    MICE-impute the cost and QALY columns, return analysis-ready DataFrame.
    """
    # Extract randomization data (needed by extract_demographics)
    pax_random = extract_randomization_data(df_raw_for_demo)

    # Get demographics
    demo_df = extract_demographics(df_raw_for_demo, pax_random)

    # Merge
    df = df_step1.merge(demo_df, on='participant_id', how='left')
    df = df[df['rand_group'].isin(GROUPS)].copy()

    # Compute cost and QALY before imputation (they may contain NaN)
    df['total_cost_28d'] = compute_total_cost_28d(df)
    df['qaly28_val']     = get_qaly_28d(df)   # copy for imputation target

    # Columns to impute: cost components + QALY
    cost_cols = [
        'day14_fam_dr', 'day14_walkin', 'day14_tel_health', 'day14_er', 'day14_hospital',
        'fup_cov_oupat_visit_numb', 'fup_cov_er_visit_numb', 'fup_cov_hosp_numb',
        'fup28_cov_oupat_visit_numb', 'fup28_cov_er_visit_numb', 'fup28_cov_hosp_numb',
        'eq5d_utility_baseline', 'eq5d_utility_21', 'eq5d_utility_28', 'qaly28', 'total_cost_28d',
    ]
    demo_cols_num = ['age', 'sex', 'race'] ##'food_security_score'
    impute_cols   = [c for c in cost_cols + demo_cols_num if c in df.columns]

    print(f"  Running MICE on {len(impute_cols)} columns for {len(df)} participants...")
    df_imp = mice_impute(df, impute_cols)

    # Re-compute total_cost_28d after imputation (component cols updated)
    df_imp['total_cost_28d'] = compute_total_cost_28d(df_imp)
    # Re-compute qaly28 after imputation
    df_imp['qaly28'] = (
        (df_imp['eq5d_utility_baseline'] + df_imp['eq5d_utility_21']) * 21
        + (df_imp['eq5d_utility_21'] + df_imp['eq5d_utility_28']) * 7
    ) / (2 * 365)
    # Drop temporary column
    df_imp = df_imp.drop(columns=['qaly28_val'], errors='ignore')

    # Round
    num_cols = df_imp.select_dtypes(include=[np.number]).columns
    df_imp[num_cols] = df_imp[num_cols].round(3)

    return df_imp


# ---------------------------------------------------------------------------
# 5. Table 6 – Cost-effectiveness results (base case + sub-analyses)
# ---------------------------------------------------------------------------

def _one_cea_row(df_sub: pd.DataFrame, label: str,
                 wtp_thresholds: list) -> dict:
    """
    Compute one row of Table 6 for a given (filtered) dataset.
    Uses total_cost_28d and qaly28.
    """
    df_sub = df_sub[df_sub['rand_group'].isin(GROUPS)].copy()
    pax = df_sub[df_sub['rand_group'] == 'Paxlovid']
    uc  = df_sub[df_sub['rand_group'] == 'Usual Care']

    cost_pax = pd.to_numeric(pax['total_cost_28d'], errors='coerce').dropna().values
    cost_uc  = pd.to_numeric(uc['total_cost_28d'],  errors='coerce').dropna().values
    qaly_pax = pd.to_numeric(pax['qaly28'], errors='coerce').dropna().values
    qaly_uc  = pd.to_numeric(uc['qaly28'],  errors='coerce').dropna().values

    n_pax = len(cost_pax)
    n_uc  = len(cost_uc)
    if n_pax < 2 or n_uc < 2:
        return {'Analysis': label}

    inc_cost = cost_pax.mean() - cost_uc.mean()
    inc_qaly = qaly_pax.mean() - qaly_uc.mean()
    icer     = inc_cost / inc_qaly if inc_qaly != 0 else np.nan

    rng = np.random.default_rng(SEED)
    boot_icer = []
    boot_inb  = {wtp: [] for wtp in wtp_thresholds}

    for _ in range(N_BOOT):
        bc_pax = rng.choice(cost_pax, n_pax, replace=True).mean()
        bc_uc  = rng.choice(cost_uc,  n_uc,  replace=True).mean()
        bq_pax = rng.choice(qaly_pax, n_pax, replace=True).mean()
        bq_uc  = rng.choice(qaly_uc,  n_uc,  replace=True).mean()
        b_inc_cost = bc_pax - bc_uc
        b_inc_qaly = bq_pax - bq_uc
        b_icer = b_inc_cost / b_inc_qaly if b_inc_qaly != 0 else np.nan
        boot_icer.append(b_icer)
        for wtp in wtp_thresholds:
            boot_inb[wtp].append(b_inc_qaly * wtp - b_inc_cost)

    icer_ci_lo = round(np.nanpercentile(boot_icer, 2.5), 3)
    icer_ci_hi = round(np.nanpercentile(boot_icer, 97.5), 3)

    cost_lo, cost_hi = _boot_ci_diff(pax['total_cost_28d'], uc['total_cost_28d'])
    qaly_lo, qaly_hi = _boot_ci_diff(pax['qaly28'],         uc['qaly28'])

    row = {
        'Analysis':                label,
        'Pax:Usual Sample Size':   f"{n_pax}:{n_uc}",
        'Mean Cost: Pax (SE)':     _mean_se(pax['total_cost_28d']),
        'Mean Cost: Usual (SE)':   _mean_se(uc['total_cost_28d']),
        'Incremental Cost (95%CI)':f"{inc_cost:.3f} {_ci_str(cost_lo, cost_hi)}",
        'Mean QALY: Pax (SE)':     _mean_se(pax['qaly28']),
        'Mean QALY: Usual (SE)':   _mean_se(uc['qaly28']),
        'Incremental QALY (95%CI)':f"{inc_qaly:.3f} {_ci_str(qaly_lo, qaly_hi)}",
        'ICER/QALY':               f"{icer:.3f} ({icer_ci_lo:.3f}, {icer_ci_hi:.3f})",
    }
    for wtp in wtp_thresholds:
        nb_arr = np.array(boot_inb[wtp])
        prob   = round(float((nb_arr > 0).mean()), 3)
        row[f'Prob CE at WTP={wtp:,}'] = prob
    return row


def table6_cea(df_imputed: pd.DataFrame, df_full: pd.DataFrame,
               out_csv: str = None,
               wtp_thresholds: list = None) -> pd.DataFrame:
    """
    Table 6: Cost-effectiveness results summary.

    Rows:
      1. Base case (MICE imputed full sample)
      2. Sensitivity: complete cases only (no missing before imputation)
      3. Subgroup: age >= 75
      4. Subgroup: age >= 65
      5. Subgroup: multimorbidity >= 2
      6. Subgroup: multimorbidity >= 3

    Parameters
    ----------
    df_imputed  : imputed analysis dataset from build_analysis_dataset()
    df_full     : same analysis dataset BEFORE imputation (for complete-case sensitivity)
    out_csv     : output path
    wtp_thresholds : list of WTP thresholds; default [30000, 50000, 70000]
    """
    if out_csv is None:
        out_csv = os.path.join(RESULTS_DIR, 'table6_cea.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if wtp_thresholds is None:
        wtp_thresholds = [30_000, 50_000, 70_000]

    rows = []

    # 1. Base case (imputed full sample)
    rows.append(_one_cea_row(df_imputed, 'Base case (MICE imputed)', wtp_thresholds))

    # 2. Sensitivity: complete cases (no NaN in cost/QALY key cols before imputation)
    cc_cols = ['total_cost_28d', 'qaly28']
    cc_mask = df_full[[c for c in cc_cols if c in df_full.columns]].notna().all(axis=1)
    df_cc   = df_full[cc_mask].copy()
    print(f"  Complete cases: {cc_mask.sum()} / {len(df_full)}")
    rows.append(_one_cea_row(df_cc, 'Sensitivity: complete cases', wtp_thresholds))

    # 3. Subgroup: age >= 75  (from imputed dataset)
    if 'age' in df_imputed.columns:
        df_75 = df_imputed[pd.to_numeric(df_imputed['age'], errors='coerce') >= 75].copy()
        print(f"  Age>=75 subgroup: n={len(df_75)}")
        rows.append(_one_cea_row(df_75, 'Subgroup: age >= 75', wtp_thresholds))

    # 4. Subgroup: age >= 65
    if 'age' in df_imputed.columns:
        df_65 = df_imputed[pd.to_numeric(df_imputed['age'], errors='coerce') >= 65].copy()
        print(f"  Age>=65 subgroup: n={len(df_65)}")
        rows.append(_one_cea_row(df_65, 'Subgroup: age >= 65', wtp_thresholds))

    # 5. Subgroup: multimorbidity >= 2
    if 'multimorbidity_ge2' in df_imputed.columns:
        df_mm2 = df_imputed[
            pd.to_numeric(df_imputed['multimorbidity_ge2'], errors='coerce') == 1
        ].copy()
        print(f"  Multimorbidity>=2 subgroup: n={len(df_mm2)}")
        rows.append(_one_cea_row(df_mm2, 'Subgroup: multimorbidity >= 2', wtp_thresholds))

    # 6. Subgroup: multimorbidity >= 3
    if 'multimorbidity_ge3' in df_imputed.columns:
        df_mm3 = df_imputed[
            pd.to_numeric(df_imputed['multimorbidity_ge3'], errors='coerce') == 1
        ].copy()
        print(f"  Multimorbidity>=3 subgroup: n={len(df_mm3)}")
        rows.append(_one_cea_row(df_mm3, 'Subgroup: multimorbidity >= 3', wtp_thresholds))

    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"Table 6 saved → {out_csv}")
    return result


# ---------------------------------------------------------------------------
# 6. Net-benefit regression
# ---------------------------------------------------------------------------

def net_benefit_regression(df: pd.DataFrame, wtp: float = 50_000,
                            out_csv: str = None) -> pd.DataFrame:
    """
    Net-benefit regression:
        NB = qaly28 * wtp - total_cost_28d

    Covariates:
      - rand_group (paxlovid dummy)
      - age (z-score)
      - sex: female vs male (exclude intersex)
      - race: non-white vs white binary
      - income: binary <=2 (<=39 999) vs >2

    Returns a DataFrame of regression coefficients.
    """
    if out_csv is None:
        out_csv = os.path.join(RESULTS_DIR, 'table6b_nb_regression.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = df[df['rand_group'].isin(GROUPS)].copy()
    df['net_benefit'] = (
        pd.to_numeric(df['qaly28'],          errors='coerce') * wtp
        - pd.to_numeric(df['total_cost_28d'], errors='coerce')
    )
    df['paxlovid'] = (df['rand_group'] == 'Paxlovid').astype(int)

    # Age (z-score)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        m, s = df['age'].mean(), df['age'].std()
        df['age_z'] = (df['age'] - m) / s if s > 0 else 0.0

    # Sex: 1=Male, 2=Female, 3=Intersex → female dummy, exclude intersex
    sex_num = pd.to_numeric(df['sex'], errors='coerce')
    df = df[sex_num.isin([1, 2])].copy()    # exclude intersex
    sex_num = pd.to_numeric(df['sex'], errors='coerce')
    df['sex_female'] = (sex_num == 2).astype(float)

    # Race: white (1) vs non-white
    race_num = pd.to_numeric(df['race'], errors='coerce')
    df['race_nonwhite'] = (race_num != 1).astype(float)
    df.loc[race_num.isna(), 'race_nonwhite'] = np.nan

    # Income: <=2 (<=39 999) vs >2
    income_num = pd.to_numeric(df['income'], errors='coerce')
    df['income_low'] = (income_num <= 2).astype(float)
    df.loc[income_num.isna(), 'income_low'] = np.nan

    X_cols = ['paxlovid', 'age_z', 'sex_female', 'race_nonwhite', 'income_low']
    X_cols = [c for c in X_cols if c in df.columns]

    analysis_df = df[['net_benefit'] + X_cols].dropna()
    if len(analysis_df) < 10:
        print("  Not enough complete cases for regression.")
        return pd.DataFrame()

    X = sm.add_constant(analysis_df[X_cols])
    y = analysis_df['net_benefit']

    model = sm.OLS(y, X).fit()

    result = pd.DataFrame({
        'predictor':  model.params.index,
        'coef':       model.params.round(3).values,
        'SE':         model.bse.round(3).values,
        'p_value':    model.pvalues.round(3).values,
        'CI_lower':   model.conf_int()[0].round(3).values,
        'CI_upper':   model.conf_int()[1].round(3).values,
    })
    result['WTP'] = wtp
    result['n']   = len(analysis_df)
    result['R2']  = round(model.rsquared, 3)

    result.to_csv(out_csv, index=False)
    print(f"Net-benefit regression (WTP={wtp:,}) saved → {out_csv}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=== STEP 3: Cost-Effectiveness Analysis ===\n")

    # ── load step-1 data ──
    print("Building Step 1 datasets...")
    from step1_data_preparation import build_step1_dataset, load_and_prepare_data, combine_fpp_into_pdd
    df_step1, df_baseline = build_step1_dataset()
    df_raw = load_and_prepare_data()
    df_raw = combine_fpp_into_pdd(df_raw)
    print(f"  step1 shape: {df_step1.shape}")
    print(f"  Groups: {df_step1['rand_group'].value_counts().to_dict()}\n")

    # ── build pre-imputation dataset (for complete-case sensitivity) ──
    print("Building pre-imputation dataset (for complete-case sensitivity)...")
    pax_random = extract_randomization_data(df_raw)
    demo_df    = extract_demographics(df_raw, pax_random)
    df_pre_imp = df_step1.merge(demo_df, on='participant_id', how='left')
    df_pre_imp = df_pre_imp[df_pre_imp['rand_group'].isin(GROUPS)].copy()
    df_pre_imp['total_cost_28d'] = compute_total_cost_28d(df_pre_imp)

    # ── build imputed analysis dataset ──
    print("Building imputed analysis dataset (demographics + cost + QALY + MICE)...")
    df_analysis = build_analysis_dataset(df_step1, df_raw)

    out_analysis = os.path.join(RESULTS_DIR, 'step3_analysis_dataset.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_analysis.to_csv(out_analysis, index=False)
    print(f"  Analysis dataset saved → {out_analysis}")
    print(f"  Shape: {df_analysis.shape}\n")

    # ── Table 6 ──
    print("── Table 6: Cost-effectiveness results ──")
    t6 = table6_cea(df_analysis, df_pre_imp)
    print(t6.to_string(index=False))

    # ── Net-benefit regression ──
    print("\n── Net-benefit regression (WTP = $50,000) ──")
    nb_reg = net_benefit_regression(df_analysis, wtp=50_000)
    if not nb_reg.empty:
        print(nb_reg.to_string(index=False))

    # Additional WTP thresholds for NB regression
    for wtp_val in [30_000, 70_000]:
        out = os.path.join(RESULTS_DIR, f'table6b_nb_regression_wtp{wtp_val}.csv')
        net_benefit_regression(df_analysis, wtp=wtp_val, out_csv=out)
