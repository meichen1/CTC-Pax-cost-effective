"""
Helpers for the Paxlovid cost-effectiveness analysis.

"""

import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
# Override with the CTC_DATA_DIR environment variable if the data lives elsewhere.
DATA_DIR = os.environ.get('CTC_DATA_DIR', os.path.join(HERE, '..', '..', 'data'))
RCC_CSV = os.path.join(DATA_DIR, 'CSV_COVID_RCC_Data_Export_ALL_Final_2025-05-15-180401369.csv')
ANTIOX_CSV = os.path.join(DATA_DIR, 'CSV_RCC_Data_Export_ALL_Final_2025-05-15-Antiox.csv')


def replace_empty_with_none(df):
    """Replace empty strings in the DataFrame with None."""
    if df is None:
        return df
    for col_name in df.columns:
        df[col_name] = df[col_name].replace(['', 'NA', 'N/A', 'na', 'NaN'], None)
    return df


def load_raw_data(csv_path):
    """Load raw data from a CSV file."""
    return pd.read_csv(csv_path)


def extract_randomization_data(df):
    """Extract and clean randomization data."""
    random_df = df[df['redcap_event_name'] == 'Randomization'].copy()
    random_df = random_df.dropna(axis=1, how='all')
    return replace_empty_with_none(random_df)


# ---------------------------------------------------------------------------
# Recovery and symptom-alleviation helpers used across analysis scripts


def ret_any_one(row: pd.Series):
    """Check if any value in row is 1."""
    if row.empty:
        return None
    elif 1 in row.values:
        return 1
    return 0

def ret_multimorbidity(row: pd.Series, num_comorbidities=2):
    """Check if a participant has at least a specified number of comorbidities."""
    if row.empty:
        return None
    return (row == 1).sum()>=num_comorbidities


def load_and_prepare_data():
    """Load and combine RCC and Antioxidant datasets for Paxlovid participants."""
    
    # Load RCC data
    df_rcc = pd.read_csv(RCC_CSV)
    # Load Antioxidant data using shared load_raw_data from antiox_primary_2603
    df_antiox = load_raw_data(ANTIOX_CSV)
    
    # Get the 4 paxlovid IDs that were initially randomized in antioxidant study
    pax_ids_in_antiox = (
        df_antiox[df_antiox['redcap_event_name'] == 'Randomization']
        .sort_values('rand_date')
        ['participant_id']
        .head(4)
        .tolist()
    )
    print(f"Paxlovid participant IDs in antioxidant dataset: {pax_ids_in_antiox}")
        
    expected_pax_ids = ['3-1001', '3-1002', '3-1003', '3-1004']
    if pax_ids_in_antiox != expected_pax_ids:
        raise ValueError(
            f"Expected Paxlovid IDs {expected_pax_ids}, got {pax_ids_in_antiox}"
        )
        
    # Filter antioxidant data to only these 4 participants
    df_antiox = df_antiox[df_antiox['participant_id'].isin(pax_ids_in_antiox)]
    
    # Get shared columns
    shared_columns = list(set(df_rcc.columns).intersection(set(df_antiox.columns)))
    
    # Align dtypes
    for col_name in shared_columns:
        if col_name in df_rcc.columns and col_name in df_antiox.columns:
            if df_rcc[col_name].dtype != df_antiox[col_name].dtype:
                try:
                    df_antiox[col_name] = df_antiox[col_name].astype(df_rcc[col_name].dtype)
                except Exception as e:
                    print(f"Could not convert column {col_name}: {e}")
    
    # Combine datasets
    df_pax = pd.concat([
        df_rcc[[col for col in shared_columns]], 
        df_antiox[[col for col in shared_columns]]
    ], ignore_index=True)
    
    return df_pax



def prepare_baseline_data(df_pax):
    """Prepare baseline characteristics table."""
    
    # Get randomization data
    pax_random = df_pax[df_pax['redcap_event_name'] == 'Randomization']
    pax_random = pax_random.dropna(axis=1, how='all')
    pax_random = replace_empty_with_none(pax_random)
    
    # Get baseline data
    pax_baseline = df_pax[df_pax['redcap_event_name'] == 'Baseline']
    pax_baseline = pax_baseline.dropna(axis=1, how='all')
    pax_baseline = replace_empty_with_none(pax_baseline)
    
    # Merge with randomization data
    pax_baseline = pax_baseline.merge(
        pax_random[['participant_id', 'rand_group', 'symp_onset_date', 'rand_date']], 
        on='participant_id', 
        how='left'
    )
    
    # Drop week columns
    pax_baseline = pax_baseline.loc[:, ~pax_baseline.columns.str.startswith(('wk', 'wend')) & ~pax_baseline.columns.str.contains('redcap_record_metadata')]
    
    
    # Calculate duration of symptoms
    pax_baseline['symp_onset_date'] = pd.to_datetime(pax_baseline['symp_onset_date'], errors='coerce')
    pax_baseline['visit_date'] = pd.to_datetime(pax_baseline['visit_date'], errors='coerce')
    pax_baseline['rand_date'] = pd.to_datetime(pax_baseline['rand_date'], errors='coerce')
    pax_baseline['duration_symp'] = (pax_baseline['visit_date'] - pax_baseline['symp_onset_date']).dt.days
    
    # Fix age calculation for any zero values
    if 'dem_age_calc' in pax_baseline.columns:
        pax_baseline['dem_age_calc'] = pd.to_numeric(pax_baseline['dem_age_calc'], errors='coerce')
        
        # Fix participants with age 0
        mask = pax_baseline['dem_age_calc'] == 0
        if mask.any():
            if 'dem_dob' in pax_baseline.columns:
                pax_baseline['dem_dob'] = pd.to_datetime(pax_baseline['dem_dob'], errors='coerce')
                pax_baseline.loc[mask, 'dem_age_calc'] = (
                    (pax_baseline.loc[mask, 'rand_date'] - pax_baseline.loc[mask, 'dem_dob']).dt.days // 365.2425
                )
    
    # Adjust ethnicity - map all Asian categories (6,7,8) to 4
    if 'dem_race' in pax_baseline.columns:
        pax_baseline['dem_race'] = pax_baseline['dem_race'].map({
            1: 1,   # White
            2: 2,   # Black
            3: 3,   # Latino
            4: 4,   # Asian
            5: 5,   # Indigenous
            6: 4,   # Map to Asian
            7: 4,   # Map to Asian
            8: 4,   # Map to Asian
            9: 9,   # Mixed
            99: 99  # Other
        })
    
    # Calculate any symptom rated moderate or major (2 or 3)
    symptom_cols = ['covid_fever', 'covid_cough', 'covid_sob', 'covid_taste', 
                    'covid_muscle_ache', 'covid_nausea', 'covid_fatigue1', 
                    'covid_concetrate', 'covid_mood']
    
    available_symptom_cols = [col for col in symptom_cols if col in pax_baseline.columns]
    
    if available_symptom_cols:
        pax_baseline[available_symptom_cols] = pax_baseline[available_symptom_cols].apply(
            pd.to_numeric, errors='coerce'
        )
        pax_baseline['any_symptom_rated23'] = pax_baseline[available_symptom_cols].apply(
            lambda x: 1 if x.isin([2, 3]).any() else (0 if x.notna().any() else None), 
            axis=1
        )
    
    ### Combine comorbidities into disease categories
    
    # Lung disease
    lung_cols = ['chronic_disease_list___2', 'chronic_disease_list___9']
    available_lung = [col for col in lung_cols if col in pax_baseline.columns]
    if available_lung:
        pax_baseline['lung_disease'] = pax_baseline[available_lung].apply(ret_any_one, axis=1)
    
    # Liver disease
    liver_cols = ['chronic_disease_list___5', 'chronic_disease_list___15', 'chronic_disease_list___16']
    available_liver = [col for col in liver_cols if col in pax_baseline.columns]
    if available_liver:
        pax_baseline['liver_disease'] = pax_baseline[available_liver].apply(ret_any_one, axis=1)
    
    # Neurological disease
    neuro_cols = ['chronic_disease_list___7', 'chronic_disease_list___8', 'chronic_disease_list___10',
                  'chronic_disease_list___12', 'chronic_disease_list___21']
    available_neuro = [col for col in neuro_cols if col in pax_baseline.columns]
    if available_neuro:
        pax_baseline['neurological_disease'] = pax_baseline[available_neuro].apply(ret_any_one, axis=1)
    
    # Heart disease
    heart_cols = ['chronic_disease_list___26', 'chronic_disease_list___27', 'chronic_disease_list___28',
                  'chronic_disease_list___29', 'chronic_disease_list___18']
    available_heart = [col for col in heart_cols if col in pax_baseline.columns]
    if available_heart:
        pax_baseline['heart_disease'] = pax_baseline[available_heart].apply(ret_any_one, axis=1)
    
    # Any disease
    disease_cols = [f'chronic_disease_list___{i}' for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,99]]
    
    # mental illness (diagnose_addict)
    if 'diagnose_addict' in pax_baseline.columns:
        disease_cols.append('diagnose_addict')
    available_disease = [col for col in disease_cols if col in pax_baseline.columns]
    ## if participant has any of chronic diseases, or mental illness, they are counted as having "any_disease" for comorbidity.
    if available_disease:
        pax_baseline['any_disease'] = pax_baseline[available_disease].apply(ret_any_one, axis=1)
        ## if participant has at least 2 of chronic diseases, or mental illness, they are counted as having "multimorbidity" for comorbidity.
        pax_baseline['multimorbidity_ge2'] = pax_baseline[available_disease].apply(ret_multimorbidity, axis=1)
        pax_baseline['multimorbidity_ge3'] = pax_baseline[available_disease].apply(lambda row: ret_multimorbidity(row, num_comorbidities=3), axis=1)
    return pax_baseline



def summarize_baseline(df, out_csv=os.path.join(HERE, 'results_cost_pax', 'paxlovid_baseline_table1.csv')):
    """
    Build Table 1-style summary for baseline characteristics.
    Compares Paxlovid vs Usual Care.
    """
    
    if 'rand_group' not in df.columns:
        print("ERROR: rand_group column not found")
        return None
    
    # FIX 3: Filter to only Paxlovid and Usual Care
    df = df[df['rand_group'].isin(['Paxlovid', 'Usual Care'])].copy()
    
    groups = ['Paxlovid', 'Usual Care']
    
    # Helper to compute n and percent
    def n_pct(sub, val=None):
        if val is None:
            count = sub.notna().sum()
        else:
            count = (sub == val).sum()
        total = sub.notna().sum()
        pct = 100 * count / total if total > 0 else 0
        return f"{count}/{total} ({pct:.3f}%)"  # FIX 3: 3 decimal places

    def ttest_pvalue(series):
        pax = pd.to_numeric(series[df['rand_group'] == 'Paxlovid'], errors='coerce').dropna()
        uc = pd.to_numeric(series[df['rand_group'] == 'Usual Care'], errors='coerce').dropna()
        if len(pax) < 2 or len(uc) < 2:
            return np.nan
        return round(stats.ttest_ind(pax, uc, equal_var=False).pvalue, 3)

    def mannwhitney_pvalue(series):
        pax = pd.to_numeric(series[df['rand_group'] == 'Paxlovid'], errors='coerce').dropna()
        uc = pd.to_numeric(series[df['rand_group'] == 'Usual Care'], errors='coerce').dropna()
        if len(pax) < 2 or len(uc) < 2:
            return np.nan
        return round(stats.mannwhitneyu(pax, uc, alternative='two-sided').pvalue, 3)

    def chi_square_pvalue(series):
        try:
            table = pd.crosstab(df['rand_group'], series)
            print(table)
            if table.shape[0] < 2 or table.shape[1] < 2:
                return np.nan
            return round(stats.chi2_contingency(table).pvalue, 3)
        except Exception:
            return np.nan
    
    # Record which columns we use
    used = {
        'age': 'dem_age_calc',
        'sex': 'dem_sex',
        'gender': 'dem_sex_identity',
        'ethnicity': 'dem_race',
        'duration_symptoms': 'duration_symp',
        'vax_doses': 'dem_vaccination_status',
        'bmi': 'bmi',
        'fever': 'covid_fever',
        'cough': 'covid_cough',
        'sob': 'covid_sob',
        'loss_smell': 'covid_taste',
        'muscle_ache': 'covid_muscle_ache',
        'nausea': 'covid_nausea',
        'fatigue': 'covid_fatigue1',
        'concentration': 'covid_concetrate',
        'anxious': 'covid_mood',
        'any_symptom_rated_moderate_or_major': 'any_symptom_rated23',
        'lung_disease': 'lung_disease',
        'liver_disease': 'liver_disease',
        'neurological_disease': 'neurological_disease',
        'heart_disease': 'heart_disease',
        'diabetes': 'chronic_disease_list___11',
        'hypertension': 'chronic_disease_list___19',
        'kidney_disease': 'chronic_disease_list___20',
        'obesity': 'chronic_disease_list___22',
        'mental_illness': 'diagnose_addict',
        'any_disease': 'any_disease',
        # Fix 2.1: Add demographics
        'education': 'dem_education',
        'income': 'dem_house_income',
        'employment': 'dem_empl_status',
        'housing': 'dem_housing',
        # Fix 2.3/2.4: Corrected food security variable names from codebook
        'food_worry': 'house_food',
        'food_last': 'house__food_last',
        'food_balanced': 'house_meals',
        'food_cut': 'house_meals_size',
        'food_cut_freq': 'house_meals_cut_often',
        'food_eat_less': 'house_eat_less',
        'food_hungry': 'house_hungry'
    }
    
    # Build rows
    rows = []
    overall_label = "Overall"
    
    def add_row_scalar(label, func, pval=None, test_name=''):
        row = {'metric': label}
        for g in groups:
            sub = df[df['rand_group'] == g]
            row[g] = func(sub)
        row[overall_label] = func(df)
        row['p_value'] = pval
        row['test'] = test_name
        rows.append(row)

    def add_missing_row(col):
        """Add a Missing row: count and % of rows with NaN in col (% = missing/total rows)."""
        total_overall = len(df)
        missing_overall = int(df[col].isna().sum())
        pct_overall = 100 * missing_overall / total_overall if total_overall > 0 else 0
        row = {'metric': '  Missing'}
        for g in groups:
            sub = df[df['rand_group'] == g]
            total_g = len(sub)
            missing_g = int(sub[col].isna().sum())
            pct_g = 100 * missing_g / total_g if total_g > 0 else 0
            row[g] = f"{missing_g}/{total_g} ({pct_g:.3f}%)"
        row[overall_label] = f"{missing_overall}/{total_overall} ({pct_overall:.3f}%)"
        row['p_value'] = ''
        rows.append(row)

    # Age
    age_col = used.get('age')
    if age_col and age_col in df.columns:
        df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
        
        def age_mean_sd(sub):
            vals = sub[age_col].dropna()
            if len(vals) > 0:
                return f"{vals.mean():.3f} ({vals.std():.3f})"  # FIX 3: 3 decimal places
            return "N/A"
        
        add_row_scalar("Age, mean (SD)", age_mean_sd, ttest_pvalue(df[age_col]), test_name='t-test')
        add_missing_row(age_col)

    # Sex
    sex_col = used.get('sex')
    if sex_col and sex_col in df.columns:
        rows.append({'metric': 'Sex, n (%)', 'p_value': chi_square_pvalue(df[sex_col]), 'test': 'chi-square'})
        
        sex_mapping = {1: 'Male', 2: 'Female', 3: 'Intersex'}
        
        for sex_val, sex_label in sex_mapping.items():
            row = {'metric': f"  {sex_label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub[sex_col], sex_val)
            row[overall_label] = n_pct(df[sex_col], sex_val)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row(sex_col)

    # Gender
    gender_col = used.get('gender')
    if gender_col and gender_col in df.columns:
        rows.append({'metric': 'Gender Identity, n (%)', 'p_value': chi_square_pvalue(df[gender_col]), 'test': 'chi-square'})
        gender_mapping = {1: 'Woman', 2: 'Man', 3: 'Gender fluid/non-binary', 4: 'Two-spirit', 99: 'Other'} 
        for gender_val, gender_label in gender_mapping.items():
            row = {'metric': f"  {gender_label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub[gender_col], gender_val)
            row[overall_label] = n_pct(df[gender_col], gender_val)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row(gender_col)

    # Ethnicity
    eth_col = used.get('ethnicity')
    if eth_col and eth_col in df.columns:
        rows.append({'metric': 'Ethnicity, n (%)', 'p_value': chi_square_pvalue(df[eth_col]), 'test': 'chi-square'})
        
        eth_mapping = {1: 'White', 2: 'Black', 3: 'Latino', 4: 'Asian', 5: 'Indigenous', 9: 'Mixed', 99: 'Other'}
        
        for eth_val, eth_label in eth_mapping.items():
            row = {'metric': f"  {eth_label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub[eth_col], eth_val)
            row[overall_label] = n_pct(df[eth_col], eth_val)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row(eth_col)

    # Duration of symptoms
    dur_col = used.get('duration_symptoms')
    if dur_col and dur_col in df.columns:
        df[dur_col] = pd.to_numeric(df[dur_col], errors='coerce')
        
        def dur_mean_sd(sub):
            vals = sub[dur_col].dropna()
            if len(vals) > 0:
                return f"{vals.mean():.3f} ({vals.std():.3f})"  # FIX 3: 3 decimal places
            return "N/A"
        
        def dur_median_iqr(sub):
            vals = sub[dur_col].dropna()
            if len(vals) > 0:
                q25, q50, q75 = vals.quantile([0.25, 0.5, 0.75])
                return f"{q50:.3f} ({q25:.3f}-{q75:.3f})"  # FIX 3: 3 decimal places
            return "N/A"
        
        add_row_scalar("Duration of symptoms (days), mean (SD)", dur_mean_sd, ttest_pvalue(df[dur_col]), test_name='t-test')
        add_row_scalar("Duration of symptoms (days), median (IQR)", dur_median_iqr, '')
        add_missing_row(dur_col)

    # Vaccine doses
    vax_col = used.get('vax_doses')
    if vax_col and vax_col in df.columns:
        rows.append({'metric': 'COVID-19 vaccine doses, n (%)', 'p_value': chi_square_pvalue(df[vax_col]), 'test': 'chi-square'})
        
        vax_mapping = {0: 'None', 1: '1 dose', 2: '2 and more doses'}
        
        for vax_val, vax_label in vax_mapping.items():
            row = {'metric': f"  {vax_label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub[vax_col], vax_val)
            row[overall_label] = n_pct(df[vax_col], vax_val)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row(vax_col)

    # BMI
    bmi_col = used.get('bmi')
    if bmi_col and bmi_col in df.columns:
        df[bmi_col] = pd.to_numeric(df[bmi_col], errors='coerce')
        
        def bmi_mean_sd(sub):
            vals = sub[bmi_col].dropna()
            if len(vals) > 0:
                return f"{vals.mean():.3f} ({vals.std():.3f})"  # FIX 3: 3 decimal places
            return "N/A"
        
        add_row_scalar("BMI, mean (SD)", bmi_mean_sd, ttest_pvalue(df[bmi_col]), test_name='t-test')
        add_missing_row(bmi_col)

    # Baseline symptoms
    rows.append({'metric': 'Baseline COVID-19 symptoms, n (%)', 'p_value': ''})
    
    symptom_list = [
        ('fever', 'Fever'),
        ('cough', 'Cough'),
        ('sob', 'Shortness of breath'),
        ('loss_smell', 'Loss of taste/smell'),
        ('muscle_ache', 'Muscle ache'),
        ('nausea', 'Nausea'),
        ('fatigue', 'Fatigue'),
        ('concentration', 'Difficulty concentrating'),
        ('anxious', 'Anxious/depressed mood'),
        ('any_symptom_rated_moderate_or_major', 'Any symptom rated moderate/major')
    ]
    
    for key, label in symptom_list:
        col = used.get(key)
        if col and col in df.columns:
            row = {'metric': f"  {label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                # Symptoms rated 1,2,3 (any present)
                if key == 'any_symptom_rated_moderate_or_major':
                    row[g] = n_pct(sub[col], 1)
                else:
                    count = sub[col].isin([1, 2, 3]).sum()
                    total = sub[col].notna().sum()
                    pct = 100 * count / total if total > 0 else 0
                    row[g] = f"{count}/{total} ({pct:.3f}%)"  # FIX 3: 3 decimal places
            
            # Overall
            if key == 'any_symptom_rated_moderate_or_major':
                row[overall_label] = n_pct(df[col], 1)
            else:
                count = df[col].isin([1, 2, 3]).sum()
                total = df[col].notna().sum()
                pct = 100 * count / total if total > 0 else 0
                row[overall_label] = f"{count}/{total} ({pct:.3f}%)"  # FIX 3: 3 decimal places
            row['p_value'] = chi_square_pvalue((df[col].isin([1, 2, 3]).astype(int) if key != 'any_symptom_rated_moderate_or_major' else df[col]))
            row['test'] = 'chi-square'
            rows.append(row)
            add_missing_row(col)

    # Comorbidities
    rows.append({'metric': 'Comorbidities, n (%)', 'p_value': ''})
    
    comorbidity_list = [
        ('diabetes', 'Diabetes'),
        ('hypertension', 'Hypertension'),
        ('kidney_disease', 'Kidney disease'),
        ('obesity', 'Obesity'),
        ('lung_disease', 'Lung disease'),
        ('liver_disease', 'Liver disease'),
        ('neurological_disease', 'Neurological disease'),
        ('heart_disease', 'Heart disease'),
        ('mental_illness', 'Mental illness'),
        ('any_disease', 'Any comorbidity')
    ]
    
    for key, label in comorbidity_list:
        col = used.get(key)
        if col and col in df.columns:
            row = {'metric': f"  {label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub[col], 1)
            row[overall_label] = n_pct(df[col], 1)
            row['p_value'] = chi_square_pvalue(df[col])
            row['test'] = 'chi-square'
            rows.append(row)
            add_missing_row(col)

    # FIX 2.1: Add demographics section
    # FIX 3-1: Combine education categories
    edu_col = used.get('education')
    if edu_col and edu_col in df.columns:
        edu_mapping = {
            1: 'High school diploma',
            2: 'High school diploma',
            3: 'College/trade/university diploma',
            4: 'College/trade/university diploma',
            5: 'Master/Doctoral degree',
            6: 'Master/Doctoral degree'
        }
        df['education_combined'] = df[edu_col].map(edu_mapping)
        edu_order = {'High school diploma': 1, 'College/trade/university diploma': 2, 'Master/Doctoral degree': 3}
        rows.append({'metric': 'Education, n (%)', 'p_value': mannwhitney_pvalue(df['education_combined'].map(edu_order)), 'test': 'Mann-Whitney U'})
        for edu_label in edu_order.keys():
            row = {'metric': f"  {edu_label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub['education_combined'], edu_label)
            row[overall_label] = n_pct(df['education_combined'], edu_label)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row('education_combined')

    # FIX 3-1: Household Income (combined)
    income_col = used.get('income')
    if income_col and income_col in df.columns:
        income_mapping = {
            1: 'Less than $39,999',
            2: 'Less than $39,999',
            3: '$40,000-$79,999',
            4: '$40,000-$79,999',
            5: '$80,000-$99,999',
            6: '$100,000 or more'
        }
        df['income_combined'] = df[income_col].map(income_mapping)
        income_order = {'Less than $39,999': 1, '$40,000-$79,999': 2, '$80,000-$99,999': 3, '$100,000 or more': 4}
        rows.append({'metric': 'Household Income, n (%)', 'p_value': mannwhitney_pvalue(df['income_combined'].map(income_order)), 'test': 'Mann-Whitney U'})
        for income_label in income_order.keys():
            row = {'metric': f"  {income_label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub['income_combined'], income_label)
            row[overall_label] = n_pct(df['income_combined'], income_label)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row('income_combined')

    # Employment Status (combined)
    emp_col = used.get('employment')
    if emp_col and emp_col in df.columns:
        emp_mapping = {
            1: 'Unemployed/seeking employment',
            2: 'Employed (full-time including self-employed)',
            3: 'Employed (part-time including self-employed)',
            4: 'Other',
            5: 'Unemployed/seeking employment',
            6: 'Unemployed/seeking employment',
            7: 'Unemployed/seeking employment',
            8: 'Unemployed/seeking employment',
            9: 'Unable to work/disabled',
            10: 'Other'
        }
        df['employment_combined'] = df[emp_col].map(emp_mapping)
        emp_order = [
            'Employed (full-time including self-employed)',
            'Employed (part-time including self-employed)',
            'Unemployed/seeking employment',
            'Unable to work/disabled',
            'Other'
        ]
        rows.append({'metric': 'Employment Status, n (%)', 'p_value': chi_square_pvalue(df['employment_combined']), 'test': 'chi-square'})
        for emp_label in emp_order:
            row = {'metric': f"  {emp_label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub['employment_combined'], emp_label)
            row[overall_label] = n_pct(df['employment_combined'], emp_label)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row('employment_combined')

    # Housing Status (combined)
    housing_col = used.get('housing')
    if housing_col and housing_col in df.columns:
        housing_mapping = {
            1: 'Other',
            2: 'Own home',
            3: 'Other',
            4: 'Other',
            5: 'Other',
            6: 'Other',
            7: 'Other',
            99: 'Other'
        }
        df['housing_combined'] = df[housing_col].map(housing_mapping)
        housing_order = ['Own home', 'Other']
        rows.append({'metric': 'Housing Status, n (%)', 'p_value': chi_square_pvalue(df['housing_combined']), 'test': 'chi-square'})
        for housing_label in housing_order:
            row = {'metric': f"  {housing_label}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub['housing_combined'], housing_label)
            row[overall_label] = n_pct(df['housing_combined'], housing_label)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row('housing_combined')

    # FIX 3-1: Food Security score summary and USDA categories
    def food_security_score_from_row(row):
        score = 0
        val = row.get('house_food')
        if val is not None and not pd.isna(val) and int(val) <= 2:
            score += 1
        val = row.get('house__food_last')
        if val is not None and not pd.isna(val) and int(val) <= 2:
            score += 1
        val = row.get('house_meals')
        if val is not None and not pd.isna(val) and int(val) <= 2:
            score += 1
        val = row.get('house_meals_size')
        if val is not None and not pd.isna(val) and int(val) == 1:
            score += 1
        val = row.get('house_eat_less')
        if val is not None and not pd.isna(val) and int(val) == 1:
            score += 1
        val = row.get('house_hungry')
        if val is not None and not pd.isna(val) and int(val) == 1:
            score += 1
        return score

    if all(col in df.columns for col in ['house_food', 'house__food_last', 'house_meals', 'house_meals_size', 'house_eat_less', 'house_hungry']):
        df['food_security_score'] = df.apply(food_security_score_from_row, axis=1)
        add_row_scalar("Food security score, mean (SD)", lambda sub: f"{sub['food_security_score'].mean():.3f} ({sub['food_security_score'].std():.3f})", ttest_pvalue(df['food_security_score']), test_name='t-test')
        add_row_scalar("Food security score, median (IQR)", lambda sub: f"{sub['food_security_score'].median():.3f} ({sub['food_security_score'].quantile(0.25):.3f}-{sub['food_security_score'].quantile(0.75):.3f})", '')

        def usda_category(score):
            if pd.isna(score):
                return np.nan
            if score <= 1:
                return 'High/Marginal food security'
            if score <= 4:
                return 'Low food security'
            return 'Very low food security'

        df['food_security_usda'] = df['food_security_score'].apply(usda_category)
        rows.append({'metric': 'USDA food security category, n (%)', 'p_value': chi_square_pvalue(df['food_security_usda']), 'test': 'chi-square'})
        for cat in ['High/Marginal food security', 'Low food security', 'Very low food security']:
            row = {'metric': f"  {cat}"}
            for g in groups:
                sub = df[df['rand_group'] == g]
                row[g] = n_pct(sub['food_security_usda'], cat)
            row[overall_label] = n_pct(df['food_security_usda'], cat)
            row['p_value'] = ''
            rows.append(row)
        add_missing_row('food_security_usda')

    # FIX 2.1/2.4: Food Security Questions (corrected variable names and mappings)
    rows.append({'metric': 'Food Security, n (%)', 'p_value': ''})
    
    food_security_questions = [
        ('food_worry', 'Worried food would run out', {1: 'Often true', 2: 'Sometimes true', 3: 'Never true', 4: "Don't know"}),
        ('food_last', "Food didn't last", {1: 'Often true', 2: 'Sometimes true', 3: 'Never true', 4: "Don't know"}),
        ('food_balanced', 'Could not afford balanced meals', {1: 'Often true', 2: 'Sometimes true', 3: 'Never true', 4: "Don't know"}),
        ('food_cut', 'Cut meal size or skipped meals', {1: 'Yes', 0: 'No', 2: "Don't know"}),
        ('food_cut_freq', 'How often cut meal size or skipped meals', {1: 'Almost every month', 2: 'Some months but not every month', 3: 'Only 1 or 2 months', 4: "Don't know"}),
        ('food_eat_less', 'Ate less than felt should', {1: 'Yes', 0: 'No', 2: "Don't know"}),
        ('food_hungry', 'Hungry but did not eat', {1: 'Yes', 0: 'No', 2: "Don't know"})
    ]
    
    for key, question_label, value_mapping in food_security_questions:
        col = used.get(key)
        if col and col in df.columns:
            rows.append({'metric': f"  {question_label}", 'p_value': chi_square_pvalue(df[col]), 'test': 'chi-square'})
            for val, val_label in value_mapping.items():
                row = {'metric': f"    {val_label}"}
                for g in groups:
                    sub = df[df['rand_group'] == g]
                    row[g] = n_pct(sub[col], val)
                row[overall_label] = n_pct(df[col], val)
                row['p_value'] = ''
                rows.append(row)
            add_missing_row(col)

    # Convert to DataFrame
    results_df = pd.DataFrame(rows)
    
    # Reorder columns
    col_order = ['metric'] + groups + [overall_label, 'p_value', 'test']
    results_df = results_df.reindex(columns=col_order, fill_value='')
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"\nBaseline characteristics table saved to: {out_csv}")
    
    return results_df


def calculate_food_security_score(row):
    """
    Calculate food security score (0-6) based on 6 questions.
    Affirmative responses = 1, non-affirmative = 0.
    
    USDA categories:
    - 0-1: High/Marginal food security
    - 2-4: Low food security  
    - 5-6: Very low food security
    
    Note: Using available food security variables from the dataset.
    """
    score = 0
    # house_food: worried food would run out (1=Often true, 2=Sometimes true)
    val = row.get('house_food')
    if val is not None and not pd.isna(val) and int(val) <= 2:
        score += 1
    
    # house__food_last: food didn't last (1=Often true, 2=Sometimes true)
    val = row.get('house__food_last')
    if val is not None and not pd.isna(val) and int(val) <= 2:
        score += 1
    
    val = row.get('house_meals')
    if val is not None and not pd.isna(val) and int(val) <= 2:
        score += 1
        
    val = row.get('house_meal_size')
    if val is not None and not pd.isna(val) and int(val) == 1:
        score += 1
        
    # val = row.get('house_meals_cut_often')
    # if val is not None and not pd.isna(val) and int(val) <= 2:
    #     score += 1
        
    val = row.get('house_eat_less')
    if val is not None and not pd.isna(val) and int(val) == 1:
        score += 1
    
    val = row.get('house_hungry')
    if val is not None and not pd.isna(val) and int(val) == 1:
        score += 1
    
    return score  




def extract_demographics(df, pax_random):
    """
    Extract demographic variables for each participant:
    - age, sex, gender, race
    - education level, income level
    - employment, housing status
    - food security score
    """
    
    # Get baseline/randomization data
    df_baseline = prepare_baseline_data(df)
    
    results = []
    # pid = '1-348'
    for pid in pax_random['participant_id'].unique():
        person_data = df_baseline[df_baseline['participant_id'] == pid]
        
        if len(person_data) == 0:
            # Create empty row
            results.append({
                'participant_id': pid,
                'age': np.nan,
                'sex': np.nan,
                'bmi': np.nan,
                # 'gender': np.nan,
                'race': np.nan,
                'education': np.nan,
                'income': np.nan,
                'employment': np.nan,
                'housing': np.nan,
                'food_security_score': np.nan,
                'any_chronic_cond': np.nan,
                'multimorbidity_ge2': np.nan,
                'multimorbidity_ge3': np.nan,
                'vaccination': np.nan
            })
            continue
        
        # Aggregate across baseline events (take first non-null)
        age = person_data['dem_age_calc'].dropna().iloc[0] if 'dem_age_calc' in person_data.columns and person_data['dem_age_calc'].notna().any() else np.nan
        ## sex categorical
        sex = person_data['dem_sex'].dropna().iloc[0] if 'dem_sex' in person_data.columns and person_data['dem_sex'].notna().any() else np.nan
        bmi = person_data['bmi'].dropna().iloc[0] if 'bmi' in person_data.columns and person_data['bmi'].notna().any() else np.nan
        ## race categorical
        race = person_data['dem_race'].dropna().iloc[0] if 'dem_race' in person_data.columns and person_data['dem_race'].notna().any() else np.nan
        
        education = person_data['dem_education'].dropna().iloc[0] if 'dem_education' in person_data.columns and person_data['dem_education'].notna().any() else np.nan
        income = person_data['dem_house_income'].dropna().iloc[0] if 'dem_house_income' in person_data.columns and person_data['dem_house_income'].notna().any() else np.nan
        ## employment categorical
        employment = person_data['dem_empl_status'].dropna().iloc[0] if 'dem_empl_status' in person_data.columns and person_data['dem_empl_status'].notna().any() else np.nan
        ## housing categorical
        housing = person_data['dem_housing'].dropna().iloc[0] if 'dem_housing' in person_data.columns and person_data['dem_housing'].notna().any() else np.nan
        any_chronic_cond = person_data['any_disease'].dropna().iloc[0] if 'any_disease' in person_data.columns and person_data['any_disease'].notna().any() else np.nan
        multimorbidity_ge2 = person_data['multimorbidity_ge2'].dropna().iloc[0] if 'multimorbidity_ge2' in person_data.columns and person_data['multimorbidity_ge2'].notna().any() else np.nan
        multimorbidity_ge3 = person_data['multimorbidity_ge3'].dropna().iloc[0] if 'multimorbidity_ge3' in person_data.columns and person_data['multimorbidity_ge3'].notna().any() else np.nan
        vaccination = person_data['dem_vaccination_status'].dropna().iloc[0] if 'dem_vaccination_status' in person_data.columns and person_data['dem_vaccination_status'].notna().any() else np.nan

        # Calculate food security score
        fs_score = np.nan
        fs_row = person_data.iloc[0].to_dict()
        fs_score = calculate_food_security_score(fs_row)
        
        # fs_row.get('house__food_last')
        
        results.append({
            'participant_id': pid,
            'age': age,
            'sex': sex,
            'bmi': bmi,
            # 'gender': gender,
            'race': race,
            'education': education,
            'income': income,
            'employment': employment,
            'housing': housing,
            'food_security_score': fs_score,
            'any_chronic_cond': any_chronic_cond,
            'multimorbidity_ge2': multimorbidity_ge2,
            'multimorbidity_ge3': multimorbidity_ge3,
            'vaccination': vaccination
        })
    
    demo_df = pd.DataFrame(results)
    return demo_df
