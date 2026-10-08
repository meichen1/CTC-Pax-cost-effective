"""
STEP 2: Result Tables for Cost-Effectiveness Analysis of Paxlovid
CanTreatCOVID Study

Tables:
  Table 1 – Baseline characteristics (with EQ-5D rows)
  Table 2 – Completion rate of resource use and EQ-5D-5L sections
  Table 3 – Mean (SE) resource use per participant by treatment arm (available cases)
  Table 4 – Mean cost (SE) of resource use per participant by treatment arm
  Table 5 – Mean (SE) EQ-5D-5L index and VAS scores by treatment arm

Each table function accepts `df` (and `df_baseline` where needed) as inputs so
the same functions can be reused for subgroup and sensitivity analyses.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from step1_data_preparation import build_step1_dataset
from helper import summarize_baseline
from cea_constants import COST_PAX, COST_OUTPAT, COST_ER, COST_HOSPITAL

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results_cost_pax')

GROUPS = ['Paxlovid', 'Usual Care']


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _filter_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Paxlovid and Usual Care rows."""
    return df[df['rand_group'].isin(GROUPS)].copy()


def _mean_se_str(values, digits: int = 3) -> str:
    vals = pd.to_numeric(values, errors='coerce').dropna()
    if len(vals) == 0:
        return 'N/A'
    m  = vals.mean()
    se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return f"{m:.{digits}f} ({se:.{digits}f})"


def _mean_diff_str(g1, g2, digits: int = 3) -> str:
    v1 = pd.to_numeric(g1, errors='coerce').dropna()
    v2 = pd.to_numeric(g2, errors='coerce').dropna()
    if len(v1) == 0 or len(v2) == 0:
        return 'N/A'
    return f"{v1.mean() - v2.mean():.{digits}f}"


def _ttest_p(g1, g2) -> float:
    v1 = pd.to_numeric(g1, errors='coerce').dropna()
    v2 = pd.to_numeric(g2, errors='coerce').dropna()
    if len(v1) < 2 or len(v2) < 2:
        return np.nan
    return round(stats.ttest_ind(v1, v2, equal_var=False).pvalue, 3)


def _mannwhitney_p(g1, g2) -> float:
    v1 = pd.to_numeric(g1, errors='coerce').dropna()
    v2 = pd.to_numeric(g2, errors='coerce').dropna()
    if len(v1) < 2 or len(v2) < 2:
        return np.nan
    return round(stats.mannwhitneyu(v1, v2, alternative='two-sided').pvalue, 3)


def _bootstrap_ci_diff(g1, g2, n_boot: int = 1000, seed: int = 42):
    """Returns (lower, upper) bootstrap 95% CI for mean(g1) - mean(g2)."""
    v1 = np.array(pd.to_numeric(g1, errors='coerce').dropna(), dtype=float)
    v2 = np.array(pd.to_numeric(g2, errors='coerce').dropna(), dtype=float)
    if len(v1) < 2 or len(v2) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = [
        rng.choice(v1, len(v1), replace=True).mean()
        - rng.choice(v2, len(v2), replace=True).mean()
        for _ in range(n_boot)
    ]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return round(float(lo), 3), round(float(hi), 3)


def _ci_str(lo, hi, digits: int = 3) -> str:
    if isinstance(lo, float) and np.isnan(lo):
        return 'N/A'
    return f"({lo:.{digits}f}, {hi:.{digits}f})"


def _available_fill_numb(df_sub: pd.DataFrame, binary_col: str, numb_col: str):
    """
    Availability = binary_col is not NaN.
    Where binary_col == 0, fill numb_col with 0 (no visit → 0 visits).
    Returns (available_boolean_mask, filled_numb_series).
    """
    avail = df_sub[binary_col].notna()
    filled = df_sub[numb_col].copy()
    no_visit = avail & (df_sub[binary_col] == 0)
    filled.loc[no_visit] = 0.0
    return avail, filled


# ---------------------------------------------------------------------------
# TABLE 1 – Baseline characteristics
# ---------------------------------------------------------------------------

def table1_baseline(df_step1: pd.DataFrame, df_baseline: pd.DataFrame,
                    out_csv: str = None) -> pd.DataFrame:
    """
    Table 1: Baseline characteristics with two additional rows after Age:
      - Mean baseline EQ-5D-5L score (SE)
      - Mean baseline EQ-5D-5L VAS (SE)
    """
    if out_csv is None:
        out_csv = os.path.join(RESULTS_DIR, 'table1_baseline.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # 1) Get the standard baseline table
    base_tbl = summarize_baseline(df_baseline, out_csv=out_csv)
    if base_tbl is None:
        return None

    # 2) Compute EQ-5D statistics from df_step1
    df = _filter_groups(df_step1)

    def _eq5d_summary_row(label: str, col: str) -> dict:
        row = {'metric': label}
        for g in GROUPS:
            sub = df[df['rand_group'] == g]
            row[g] = _mean_se_str(sub[col]) if col in df.columns else 'N/A'
        row['Overall'] = _mean_se_str(df[col]) if col in df.columns else 'N/A'
        if col in df.columns:
            row['p_value'] = _ttest_p(
                df.loc[df['rand_group'] == 'Paxlovid',  col],
                df.loc[df['rand_group'] == 'Usual Care', col],
            )
        else:
            row['p_value'] = ''
        return row
        
    def _ret_missing_row(col: str) -> dict:
        """ret a Missing row: count and % of rows with NaN in col (% = missing/total rows)."""
        total_overall = len(df)
        missing_overall = int(df[col].isna().sum())
        pct_overall = 100 * missing_overall / total_overall if total_overall > 0 else 0
        row = {'metric': '  Missing'}
        for g in GROUPS:
            sub = df[df['rand_group'] == g]
            total_g = len(sub)
            missing_g = int(sub[col].isna().sum())
            pct_g = 100 * missing_g / total_g if total_g > 0 else 0
            row[g] = f"{missing_g}/{total_g} ({pct_g:.3f}%)"
        row['Overall'] = f"{missing_overall}/{total_overall} ({pct_overall:.3f}%)"
        row['p_value'] = ''
        return row


    eq5d_index_row = _eq5d_summary_row('Mean baseline EQ-5D-5L score (SE)', 'eq5d_utility_baseline')
    eq5d_score_missing_row = _ret_missing_row('eq5d_utility_baseline')
    eq5d_vas_row   = _eq5d_summary_row('Mean baseline EQ-5D-5L VAS (SE)',   'eq5d_vas')
    eq5d_vas_missing_row = _ret_missing_row('eq5d_vas')

    # 3) Insert after Age row
    # age_idx = base_tbl[base_tbl['metric'].str.contains('Age, mean', na=False)].index
    # if len(age_idx) > 0:
    #     cut = int(age_idx[-1])
    #     top    = base_tbl.iloc[: cut + 1]
    #     bottom = base_tbl.iloc[cut + 1 :]
    #     insert = pd.DataFrame([eq5d_index_row, eq5d_score_missing_row, eq5d_vas_row, eq5d_vas_missing_row])
    #     base_tbl = pd.concat([top, insert, bottom], ignore_index=True)
    # else:
    base_tbl = pd.concat(
        [base_tbl, pd.DataFrame([eq5d_index_row, eq5d_score_missing_row, eq5d_vas_row, eq5d_vas_missing_row])],
        ignore_index=True,
    )

    base_tbl.to_csv(out_csv, index=False)
    print(f"Table 1 saved → {out_csv}")
    return base_tbl


# ---------------------------------------------------------------------------
# TABLE 2 – Completion rate
# ---------------------------------------------------------------------------

def table2_completion(df: pd.DataFrame, out_csv: str = None) -> pd.DataFrame:
    """
    Table 2: Completion rate of resource use and EQ-5D-5L sections by treatment arm.
    """
    if out_csv is None:
        out_csv = os.path.join(RESULTS_DIR, 'table2_completion.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = _filter_groups(df)
    rows = []

    def _n_pct(sub: pd.DataFrame, col: str) -> str:
        if col not in sub.columns:
            return 'N/A'
        n     = int(sub[col].notna().sum())
        total = len(sub)
        pct   = 100.0 * n / total if total > 0 else 0.0
        return f"{n} ({pct:.3f}%)"

    def section_row(label: str, col: str) -> dict:
        r = {'Section': f'  {label}'}
        for g in GROUPS:
            r[g] = _n_pct(df[df['rand_group'] == g], col)
        return r

    # Sample sizes
    r = {'Section': 'Sample size (n)'}
    for g in GROUPS:
        r[g] = str(len(df[df['rand_group'] == g]))
    rows.append(r)

    # Resource use
    rows.append({'Section': 'Resource use, n (%)'})
    rows.append(section_row('1–14 day',  'day14_fam_dr'))
    rows.append(section_row('14–21 day', 'fup_cov_outpat_visit'))
    rows.append(section_row('21–28 day', 'fup28_cov_outpat_visit'))

    # EQ-5D-5L
    rows.append({'Section': 'EQ-5D-5L, n (%)'})
    rows.append(section_row('Baseline', 'eq5d_utility_baseline'))
    rows.append(section_row('21 day',   'eq5d_utility_21'))
    rows.append(section_row('28 day',   'eq5d_utility_28'))

    result = pd.DataFrame(rows).reindex(columns=['Section'] + GROUPS, fill_value='')
    result.to_csv(out_csv, index=False)
    print(f"Table 2 saved → {out_csv}")
    return result


# ---------------------------------------------------------------------------
# TABLE 3 – Mean (SE) resource use
# ---------------------------------------------------------------------------

def table3_resource_use(df: pd.DataFrame, out_csv: str = None,
                        digits: int = 4) -> pd.DataFrame:
    """
    Table 3: Mean (SE) resource use per participant by treatment arm (available cases).
    Available-case denominator is determined by the binary (non-_numb) visit column.
    Where binary = 0, the _numb column is filled with 0.
    Bootstrap 95% CI for mean difference; Mann-Whitney U p-value.
    """
    if out_csv is None:
        out_csv = os.path.join(RESULTS_DIR, 'table3_resource_use.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = _filter_groups(df)
    OUT_COLS = ['Resource category', 'Unit',
                'Paxlovid', 'Usual Care',
                'Mean difference', 'p-value', 'Bootstrap 95%CI']
    rows = []

    def _section_header(label: str, avail_col: str):
        n_pax = int(df.loc[df['rand_group'] == 'Paxlovid',  avail_col].notna().sum()) if avail_col in df.columns else len(df[df['rand_group'] == 'Paxlovid'])
        n_uc  = int(df.loc[df['rand_group'] == 'Usual Care', avail_col].notna().sum()) if avail_col in df.columns else len(df[df['rand_group'] == 'Usual Care'])
        rows.append({
            'Resource category': label, 'Unit': '',
            'Paxlovid': f'available cases (n={n_pax})',
            'Usual Care': f'n={n_uc}',
            'Mean difference': '', 'p-value': '', 'Bootstrap 95%CI': '',
        })

    def _resource_row(label: str, unit: str, pax_vals, uc_vals):
        lo, hi = _bootstrap_ci_diff(pax_vals, uc_vals)
        rows.append({
            'Resource category': f'  {label}',
            'Unit': unit,
            'Paxlovid':        _mean_se_str(pax_vals, digits),
            'Usual Care':      _mean_se_str(uc_vals,  digits),
            'Mean difference': _mean_diff_str(pax_vals, uc_vals, digits),
            'p-value':         _mannwhitney_p(pax_vals, uc_vals),
            'Bootstrap 95%CI': _ci_str(lo, hi, digits),
        })

    # ── First 14 days ────────────────────────────────────────────────
    _section_header('First 14 day', 'day14_fam_dr')
    for col, label, unit in [
        ('day14_fam_dr',   'Family doctor',  'per visit'),
        ('day14_walkin',   'Walk-in clinic', 'per visit'),
        ('day14_er',       'Emergency',      'per visit'),
        ('day14_hospital', 'Hospital',       'per day'),
    ]:
        if col in df.columns:
            avail = df[col].notna()
            _resource_row(label, unit,
                          df.loc[(df['rand_group'] == 'Paxlovid')   & avail, col],
                          df.loc[(df['rand_group'] == 'Usual Care') & avail, col])

    # ── 14–21 days ───────────────────────────────────────────────────
    _section_header('14–21 day', 'fup_cov_outpat_visit')
    for bin_col, numb_col, label, unit in [
        ('fup_cov_outpat_visit', 'fup_cov_oupat_visit_numb', 'Walk-in clinic', 'per visit'),
        ('fup_cov_er_visit',     'fup_cov_er_visit_numb',    'Emergency',      'per visit'),
        ('fup_cov_hosp',         'fup_cov_hosp_numb',        'Hospital',       'per day'),
    ]:
        if bin_col in df.columns and numb_col in df.columns:
            ap, fp = _available_fill_numb(df[df['rand_group'] == 'Paxlovid'],   bin_col, numb_col)
            au, fu = _available_fill_numb(df[df['rand_group'] == 'Usual Care'], bin_col, numb_col)
            _resource_row(label, unit, fp[ap], fu[au])

    # ── 21–28 days ───────────────────────────────────────────────────
    _section_header('21–28 day', 'fup28_cov_outpat_visit')
    for bin_col, numb_col, label, unit in [
        ('fup28_cov_outpat_visit', 'fup28_cov_oupat_visit_numb', 'Walk-in clinic', 'per visit'),
        ('fup28_cov_er_visit',     'fup28_cov_er_visit_numb',    'Emergency',      'per visit'),
        ('fup28_cov_hosp',         'fup28_cov_hosp_numb',        'Hospital',       'per day'),
    ]:
        if bin_col in df.columns and numb_col in df.columns:
            ap, fp = _available_fill_numb(df[df['rand_group'] == 'Paxlovid'],   bin_col, numb_col)
            au, fu = _available_fill_numb(df[df['rand_group'] == 'Usual Care'], bin_col, numb_col)
            _resource_row(label, unit, fp[ap], fu[au])

    result = pd.DataFrame(rows, columns=OUT_COLS)
    result.to_csv(out_csv, index=False)
    print(f"Table 3 saved → {out_csv}")
    return result


# ---------------------------------------------------------------------------
# TABLE 4 – Mean cost (SE)
# ---------------------------------------------------------------------------

def table4_costs(df: pd.DataFrame, out_csv: str = None,
                 digits: int = 3) -> pd.DataFrame:
    """
    Table 4: Mean cost (SE) per participant by treatment arm.
    Costs (CAD): Paxlovid drug $1288.88; outpatient/family-doctor/walk-in $397;
    ER $397; hospital inpatient per day $1270.
    """
    if out_csv is None:
        out_csv = os.path.join(RESULTS_DIR, 'table4_costs.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = _filter_groups(df)
    OUT_COLS = ['Resource category', 'Unit',
                'Paxlovid', 'Usual Care',
                'Mean difference', 'p-value', 'Bootstrap 95%CI']
    rows = []

    def _section_header(label: str, avail_col: str):
        n_pax = int(df.loc[df['rand_group'] == 'Paxlovid',  avail_col].notna().sum()) if avail_col in df.columns else len(df[df['rand_group'] == 'Paxlovid'])
        n_uc  = int(df.loc[df['rand_group'] == 'Usual Care', avail_col].notna().sum()) if avail_col in df.columns else len(df[df['rand_group'] == 'Usual Care'])
        rows.append({
            'Resource category': label, 'Unit': '',
            'Paxlovid': f'available cases (n={n_pax})',
            'Usual Care': f'n={n_uc}',
            'Mean difference': '', 'p-value': '', 'Bootstrap 95%CI': '',
        })

    def _cost_row(label: str, pax_counts, uc_counts, cost_per_unit: float):
        pax_cost = pd.to_numeric(pax_counts, errors='coerce').dropna() * cost_per_unit
        uc_cost  = pd.to_numeric(uc_counts,  errors='coerce').dropna() * cost_per_unit
        lo, hi   = _bootstrap_ci_diff(pax_cost, uc_cost)
        rows.append({
            'Resource category': f'  {label}',
            'Unit': 'CAD',
            'Paxlovid':        _mean_se_str(pax_cost, digits),
            'Usual Care':      _mean_se_str(uc_cost,  digits),
            'Mean difference': _mean_diff_str(pax_cost, uc_cost, digits),
            'p-value':         _mannwhitney_p(pax_cost, uc_cost),
            'Bootstrap 95%CI': _ci_str(lo, hi, digits),
        })

    # ── Paxlovid drug cost ────────────────────────────────────────────
    rows.append({
        'Resource category': 'Paxlovid drug cost',
        'Unit': 'CAD',
        'Paxlovid':        f'{COST_PAX:.{digits}f} (0.000)',
        'Usual Care':      '0.000 (0.000)',
        'Mean difference': f'{COST_PAX:.{digits}f}',
        'p-value': '', 'Bootstrap 95%CI': '',
    })

    # ── First 14 days ────────────────────────────────────────────────
    _section_header('First 14 day', 'day14_fam_dr')
    for col, label, cost in [
        ('day14_fam_dr',   'Family doctor',  COST_OUTPAT),
        ('day14_walkin',   'Walk-in clinic', COST_OUTPAT),
        ('day14_er',       'Emergency',      COST_ER),
        ('day14_hospital', 'Hospital',       COST_HOSPITAL),
    ]:
        if col in df.columns:
            avail = df[col].notna()
            _cost_row(label,
                      df.loc[(df['rand_group'] == 'Paxlovid')   & avail, col],
                      df.loc[(df['rand_group'] == 'Usual Care') & avail, col],
                      cost)

    # ── 14–21 days ───────────────────────────────────────────────────
    _section_header('14–21 day', 'fup_cov_outpat_visit')
    for bin_col, numb_col, label, cost in [
        ('fup_cov_outpat_visit', 'fup_cov_oupat_visit_numb', 'Walk-in clinic', COST_OUTPAT),
        ('fup_cov_er_visit',     'fup_cov_er_visit_numb',    'Emergency',      COST_ER),
        ('fup_cov_hosp',         'fup_cov_hosp_numb',        'Hospital',       COST_HOSPITAL),
    ]:
        if bin_col in df.columns and numb_col in df.columns:
            ap, fp = _available_fill_numb(df[df['rand_group'] == 'Paxlovid'],   bin_col, numb_col)
            au, fu = _available_fill_numb(df[df['rand_group'] == 'Usual Care'], bin_col, numb_col)
            _cost_row(label, fp[ap], fu[au], cost)

    # ── 21–28 days ───────────────────────────────────────────────────
    _section_header('21–28 day', 'fup28_cov_outpat_visit')
    for bin_col, numb_col, label, cost in [
        ('fup28_cov_outpat_visit', 'fup28_cov_oupat_visit_numb', 'Walk-in clinic', COST_OUTPAT),
        ('fup28_cov_er_visit',     'fup28_cov_er_visit_numb',    'Emergency',      COST_ER),
        ('fup28_cov_hosp',         'fup28_cov_hosp_numb',        'Hospital',       COST_HOSPITAL),
    ]:
        if bin_col in df.columns and numb_col in df.columns:
            ap, fp = _available_fill_numb(df[df['rand_group'] == 'Paxlovid'],   bin_col, numb_col)
            au, fu = _available_fill_numb(df[df['rand_group'] == 'Usual Care'], bin_col, numb_col)
            _cost_row(label, fp[ap], fu[au], cost)

    result = pd.DataFrame(rows, columns=OUT_COLS)
    result.to_csv(out_csv, index=False)
    print(f"Table 4 saved → {out_csv}")
    return result


# ---------------------------------------------------------------------------
# TABLE 5 – EQ-5D-5L index and VAS
# ---------------------------------------------------------------------------

def table5_eq5d(df: pd.DataFrame, out_csv: str = None) -> pd.DataFrame:
    """
    Table 5: Mean (SE) EQ-5D-5L index and VAS scores by treatment arm at each
    time-point (available cases). P-value from independent t-test; bootstrap 95% CI.
    """
    if out_csv is None:
        out_csv = os.path.join(RESULTS_DIR, 'table5_eq5d.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = _filter_groups(df)
    OUT_COLS = ['', 'Paxlovid', 'Usual Care',
                'Mean difference', 'p-value (t test)', 'Bootstrap 95%CI']
    rows = []

    # Sample sizes
    n_pax = len(df[df['rand_group'] == 'Paxlovid'])
    n_uc  = len(df[df['rand_group'] == 'Usual Care'])
    rows.append({
        '': 'Sample size (n)',
        'Paxlovid': str(n_pax), 'Usual Care': str(n_uc),
        'Mean difference': '', 'p-value (t test)': '', 'Bootstrap 95%CI': '',
    })

    def _eq5d_row(indent_label: str, col: str):
        if col not in df.columns:
            rows.append({
                '': indent_label, 'Paxlovid': 'N/A', 'Usual Care': 'N/A',
                'Mean difference': 'N/A', 'p-value (t test)': 'N/A', 'Bootstrap 95%CI': 'N/A',
            })
            return
        pax_v = df.loc[df['rand_group'] == 'Paxlovid',  col]
        uc_v  = df.loc[df['rand_group'] == 'Usual Care', col]
        lo, hi = _bootstrap_ci_diff(pax_v, uc_v)
        rows.append({
            '': indent_label,
            'Paxlovid':          _mean_se_str(pax_v),
            'Usual Care':        _mean_se_str(uc_v),
            'Mean difference':   _mean_diff_str(pax_v, uc_v),
            'p-value (t test)':  _ttest_p(pax_v, uc_v),
            'Bootstrap 95%CI':   _ci_str(lo, hi),
        })

    # EQ-5D-5L index score section
    rows.append({'': 'EQ-5D-5L index score',
                 'Paxlovid': '', 'Usual Care': '',
                 'Mean difference': '', 'p-value (t test)': '', 'Bootstrap 95%CI': ''})
    _eq5d_row('  Baseline', 'eq5d_utility_baseline')
    _eq5d_row('  21 day',   'eq5d_utility_21')
    _eq5d_row('  28 day',   'eq5d_utility_28')

    # EQ-5D-5L VAS section
    rows.append({'': 'EQ-5D-5L VAS',
                 'Paxlovid': '', 'Usual Care': '',
                 'Mean difference': '', 'p-value (t test)': '', 'Bootstrap 95%CI': ''})
    _eq5d_row('  Baseline', 'eq5d_vas')
    _eq5d_row('  21 day',   'eq5d_vas_21')
    _eq5d_row('  28 day',   'eq5d_vas_28')

    result = pd.DataFrame(rows, columns=OUT_COLS)
    result.to_csv(out_csv, index=False)
    print(f"Table 5 saved → {out_csv}")
    return result


# ---------------------------------------------------------------------------
# Run all tables
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    print("Building Step 1 datasets...")
    df_step1, df_baseline = build_step1_dataset()

    print(f"\nStep 1 shape: {df_step1.shape}")
    print(f"Groups: {df_step1['rand_group'].value_counts().to_dict()}")

    print("\n── Table 1: Baseline characteristics ──")
    t1 = table1_baseline(df_step1, df_baseline)

    print("\n── Table 2: Completion rates ──")
    t2 = table2_completion(df_step1)
    print(t2.to_string(index=False))

    print("\n── Table 3: Resource use (mean SE) ──")
    t3 = table3_resource_use(df_step1)
    print(t3.to_string(index=False))

    print("\n── Table 4: Costs (mean SE) ──")
    t4 = table4_costs(df_step1)
    print(t4.to_string(index=False))

    print("\n── Table 5: EQ-5D-5L scores ──")
    t5 = table5_eq5d(df_step1)
    print(t5.to_string(index=False))
