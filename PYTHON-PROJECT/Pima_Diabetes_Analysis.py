import pandas as pd
import random
from functools import reduce

def load_dataset(filepath):
    """
    Loads the Pima Indians Diabetes CSV file into a pandas DataFrame.
    Handles FileNotFoundError, PermissionError, and other IO errors gracefully.
    """
    try:
        df = pd.read_csv(filepath)
        print("=" * 65)
        print("  FILE LOADED SUCCESSFULLY")
        print("=" * 65)
        print(f"  Dataset Shape : {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns       : {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{filepath}'")
        print("  Please place 'diabetes.csv' in the same folder as this script.")
        raise
    except PermissionError:
        print(f"[ERROR] Permission denied when reading: '{filepath}'")
        print("  Check that you have read access to the file.")
        raise
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while reading the file: {e}")
        raise

def explore_data(df):
    """
    Displays first 10 rows, last 5 rows, DataFrame info,
    and filtered rows (glucose > 140 or age > 50) using for-loops.
    """
    print("\n" + "=" * 65)
    print("  OPERATION 2 : DATA EXPLORATION")
    print("=" * 65)

    print("\n--- First 10 Rows ---")
    print(df.head(10).to_string())

    print("\n--- Last 5 Rows ---")
    print(df.tail(5).to_string())

    print("\n--- DataFrame Info ---")
    df.info()

    print("\n--- Statistical Summary ---")
    print(df.describe().to_string())

    print("\n--- Rows where Glucose > 140 OR Age > 50 (via for-loop) ---")
    count = 0
    for index, row in df.iterrows():
        if row['Glucose'] > 140 or row['Age'] > 50:
            print(f"  Row {index:4d} | Glucose={row['Glucose']:6.1f} | "
                  f"Age={row['Age']:3.0f} | BMI={row['BMI']:5.1f} | "
                  f"Outcome={int(row['Outcome'])}")
            count += 1
            if count >= 20:
                print(f"  ... (showing first 20 of matching rows)")
                break
    print(f"  Total matching rows: {((df['Glucose'] > 140) | (df['Age'] > 50)).sum()}")

def handle_missing_values(df):
    """
    Identifies zero values in key medical columns as missing data.
    Replaces them with the column median calculated manually using loops.
    """
    print("\n" + "=" * 65)
    print("  OPERATION 3 : HANDLE MISSING VALUES")
    print("=" * 65)

    zero_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    print("\n--- Zero Counts Before Replacement ---")
    for col in zero_as_missing_cols:
        zero_count = (df[col] == 0).sum()
        print(f"  {col:<20s}: {zero_count} zero(s) found")

    def manual_median(values):
        """Calculates median from a list using sorting and indexing."""
        clean = sorted([v for v in values if v != 0])
        n = len(clean)
        if n == 0:
            return 0
        mid = n // 2
        if n % 2 == 1:
            return clean[mid]
        else:
            return (clean[mid - 1] + clean[mid]) / 2

    print("\n--- Replacing Zeros with Column Medians ---")
    for col in zero_as_missing_cols:
        col_values = df[col].tolist()
        median_val = manual_median(col_values)
        replaced = 0
        for i in range(len(df)):
            if df.at[i, col] == 0:
                df.at[i, col] = median_val
                replaced += 1
        print(f"  {col:<20s}: median = {median_val:.4f}  |  replaced {replaced} zero(s)")

    print("\n--- Zero Counts After Replacement ---")
    for col in zero_as_missing_cols:
        zero_count = (df[col] == 0).sum()
        if zero_count == 0:
            print(f"  {col:<20s}: No zeros remaining ✓")
        else:
            print(f"  {col:<20s}: {zero_count} zeros still present (may be valid)")

    print("\n--- NaN Count per Column ---")
    nan_counts = df.isnull().sum()
    for col, cnt in nan_counts.items():
        status = "✓ Clean" if cnt == 0 else f"⚠ {cnt} NaN(s)"
        print(f"  {col:<25s}: {status}")

    return df

def create_new_columns(df):
    """
    Creates three new engineered feature columns using
    arithmetic operators, safe division, and if-elif-else logic.
    """
    print("\n" + "=" * 65)
    print("  OPERATION 4 : CREATE NEW COLUMNS")
    print("=" * 65)

    bmi_glucose_list = []
    for i in range(len(df)):
        bmi   = df.at[i, 'BMI']
        gluc  = df.at[i, 'Glucose']
        try:
            if gluc != 0:
                bgi = (bmi * gluc) / 100
            else:
                bgi = 0.0
        except ZeroDivisionError:
            bgi = 0.0
        bmi_glucose_list.append(round(bgi, 4))
    df['BMI_Glucose_Index'] = bmi_glucose_list

    ig_ratio_list = []
    for i in range(len(df)):
        insulin = df.at[i, 'Insulin']
        gluc    = df.at[i, 'Glucose']
        try:
            if gluc != 0:
                ratio = insulin / gluc
            else:
                ratio = 0.0
        except ZeroDivisionError:
            ratio = 0.0
        ig_ratio_list.append(round(ratio, 4))
    df['Insulin_Glucose_Ratio'] = ig_ratio_list

    age_risk_list = []
    for i in range(len(df)):
        age = df.at[i, 'Age']
        if age < 30:
            level = "Young"
        elif age >= 30 and age <= 50:
            level = "Middle-Aged"
        else:
            level = "Senior"
        age_risk_list.append(level)
    df['Age_Risk_Level'] = age_risk_list

    print("\n--- New Columns Added ---")
    print(f"  BMI_Glucose_Index      | min={df['BMI_Glucose_Index'].min():.2f}  "
          f"max={df['BMI_Glucose_Index'].max():.2f}  "
          f"mean={df['BMI_Glucose_Index'].mean():.2f}")
    print(f"  Insulin_Glucose_Ratio  | min={df['Insulin_Glucose_Ratio'].min():.4f}  "
          f"max={df['Insulin_Glucose_Ratio'].max():.4f}  "
          f"mean={df['Insulin_Glucose_Ratio'].mean():.4f}")
    print(f"  Age_Risk_Level         | categories={df['Age_Risk_Level'].value_counts().to_dict()}")

    print("\n--- Sample of New Columns (first 10 rows) ---")
    print(df[['Age', 'BMI', 'Glucose', 'Insulin',
              'BMI_Glucose_Index', 'Insulin_Glucose_Ratio',
              'Age_Risk_Level']].head(10).to_string())

    return df

def high_risk_flags(df):
    """
    Creates a High_Risk flag using logical and relational operators.
    Analyzes high-risk patients versus diabetes outcome using loops.
    """
    print("\n" + "=" * 65)
    print("  OPERATION 5 : HIGH-RISK FLAGS")
    print("=" * 65)

    high_risk_list = []
    for i in range(len(df)):
        glucose = df.at[i, 'Glucose']
        bmi     = df.at[i, 'BMI']
        age     = df.at[i, 'Age']

        is_high_risk = (glucose > 140) or (bmi > 30) or (age > 50)
        high_risk_list.append(1 if is_high_risk else 0)

    df['High_Risk'] = high_risk_list

    total_high_risk    = sum(high_risk_list)
    total_not_high_risk = len(high_risk_list) - total_high_risk

    print(f"\n  Total High-Risk Patients    : {total_high_risk}")
    print(f"  Total Non-High-Risk Patients: {total_not_high_risk}")

    hr_diabetic     = 0
    hr_non_diabetic = 0
    nhr_diabetic    = 0
    nhr_non_diabetic= 0

    for i in range(len(df)):
        hr      = df.at[i, 'High_Risk']
        outcome = df.at[i, 'Outcome']

        if hr == 1 and outcome == 1:
            hr_diabetic += 1
        elif hr == 1 and outcome == 0:
            hr_non_diabetic += 1
        elif hr == 0 and outcome == 1:
            nhr_diabetic += 1
        else:
            nhr_non_diabetic += 1

    hr_diabetes_rate  = (hr_diabetic / total_high_risk * 100) if total_high_risk != 0 else 0
    nhr_diabetes_rate = (nhr_diabetic / total_not_high_risk * 100) if total_not_high_risk != 0 else 0

    print("\n--- High-Risk vs Diabetes Outcome ---")
    print(f"  High-Risk   → Diabetic     : {hr_diabetic:4d}  |  Non-Diabetic: {hr_non_diabetic:4d}  "
          f"|  Diabetes Rate: {hr_diabetes_rate:.1f}%")
    print(f"  Not High-Risk → Diabetic   : {nhr_diabetic:4d}  |  Non-Diabetic: {nhr_non_diabetic:4d}  "
          f"|  Diabetes Rate: {nhr_diabetes_rate:.1f}%")

    safe_count = sum(1 for i in range(len(df))
                     if not (df.at[i, 'Glucose'] > 140 or
                             df.at[i, 'BMI']     > 30  or
                             df.at[i, 'Age']     > 50))
    print(f"\n  Patients with NO risk flags (safe): {safe_count}")

    return df

def insulin_glucose_ratios(df):
    """
    Uses filter() and map() from functools to calculate
    Insulin/Glucose ratio for rows where Insulin > 0.
    Compares results with manual for-loop approach.
    """
    print("\n" + "=" * 65)
    print("  OPERATION 6 : INSULIN/GLUCOSE RATIOS (filter & map)")
    print("=" * 65)

    insulin_list = df['Insulin'].tolist()
    glucose_list = df['Glucose'].tolist()
    pairs        = list(zip(insulin_list, glucose_list))

    filtered_pairs = list(filter(lambda pair: pair[0] > 0, pairs))

    ig_ratios_functional = list(map(
        lambda pair: round(pair[0] / pair[1], 4) if pair[1] != 0 else 0.0,
        filtered_pairs
    ))

    ig_ratios_manual = []
    for insulin, glucose in pairs:
        if insulin > 0:
            if glucose != 0:
                ig_ratios_manual.append(round(insulin / glucose, 4))
            else:
                ig_ratios_manual.append(0.0)

    print(f"\n  Rows with Insulin > 0 (filter)    : {len(filtered_pairs)}")
    print(f"  Rows computed via filter+map      : {len(ig_ratios_functional)}")
    print(f"  Rows computed via manual for-loop : {len(ig_ratios_manual)}")

    match = all(a == b for a, b in zip(ig_ratios_functional, ig_ratios_manual))
    print(f"  Results Match                     : {match}")

    mean_ratio = sum(ig_ratios_functional) / len(ig_ratios_functional) if ig_ratios_functional else 0
    max_ratio  = max(ig_ratios_functional) if ig_ratios_functional else 0
    min_ratio  = min(ig_ratios_functional) if ig_ratios_functional else 0

    print(f"\n  Insulin/Glucose Ratio Stats (Insulin > 0 patients):")
    print(f"    Mean   : {mean_ratio:.4f}")
    print(f"    Min    : {min_ratio:.4f}")
    print(f"    Max    : {max_ratio:.4f}")

    print(f"\n  Sample ratios (first 10) via filter+map:")
    for idx, ratio in enumerate(ig_ratios_functional[:10], 1):
        print(f"    {idx:2d}. Insulin/Glucose = {ratio:.4f}")

def recursive_total_risk(df):
    """
    Recursively accumulates total risk (BMI_Glucose_Index) across patients.
    Compares result with a simple for-loop summation.
    """
    print("\n" + "=" * 65)
    print("  OPERATION 7 : RECURSIVE SUM FOR TOTAL RISK")
    print("=" * 65)

    risk_scores = df['BMI_Glucose_Index'].tolist()
    n = len(risk_scores)

    import sys
    sys.setrecursionlimit(n + 100)

    def calculate_total_risk(index, accumulator):
        """
        Recursively sums risk scores.
        Base case: index equals number of patients → return accumulated total.
        """
        if index == n:
            return accumulator
        return calculate_total_risk(index + 1, accumulator + risk_scores[index])

    recursive_total = calculate_total_risk(0, 0.0)

    loop_total = 0.0
    for score in risk_scores:
        loop_total += score

    print(f"\n  Total Risk (Recursive sum)   : {recursive_total:.4f}")
    print(f"  Total Risk (For-loop sum)    : {loop_total:.4f}")
    print(f"  Results Match                : {abs(recursive_total - loop_total) < 0.0001}")
    print(f"  Average Risk per Patient     : {recursive_total / n:.4f}")

    hr_flags  = df['High_Risk'].tolist()

    def calculate_total_highrisk(index, accumulator):
        """Recursively counts high-risk patients."""
        if index == len(hr_flags):
            return accumulator
        return calculate_total_highrisk(index + 1, accumulator + hr_flags[index])

    total_hr_recursive = calculate_total_highrisk(0, 0)
    total_hr_loop      = sum(hr_flags)

    print(f"\n  High-Risk Count (Recursive)  : {total_hr_recursive}")
    print(f"  High-Risk Count (For-loop)   : {total_hr_loop}")
    print(f"  Results Match                : {total_hr_recursive == total_hr_loop}")

def recursive_sum(lst, index=0):
    """Recursively computes sum of a list."""
    if index == len(lst):
        return 0
    return lst[index] + recursive_sum(lst, index + 1)

def manual_mean(values):
    """Computes mean using recursive sum."""
    if not values:
        return 0
    return recursive_sum(values) / len(values)

def manual_median(values):
    """Computes median from a sorted list."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0
    mid = n // 2
    return sorted_vals[mid] if n % 2 == 1 else (sorted_vals[mid-1] + sorted_vals[mid]) / 2

def manual_mode(values):
    """Computes mode using a frequency dictionary."""
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    if not freq:
        return None
    return max(freq, key=freq.get)

def manual_min(values):
    """Finds minimum using a loop."""
    if not values:
        return None
    m = values[0]
    for v in values[1:]:
        if v < m:
            m = v
    return m

def manual_max(values):
    """Finds maximum using a loop."""
    if not values:
        return None
    m = values[0]
    for v in values[1:]:
        if v > m:
            m = v
    return m

def statistics_calculation(df):
    """
    Computes comprehensive statistics for key numeric columns
    both manually and using pandas. Groups by Outcome using dicts and sets.
    """
    print("\n" + "=" * 65)
    print("  OPERATION 8 : STATISTICS CALCULATION")
    print("=" * 65)

    stat_cols = ['Glucose', 'BMI', 'Age', 'Insulin',
                 'BloodPressure', 'Pregnancies', 'BMI_Glucose_Index']

    print("\n--- Manual Statistics (loops + recursion) ---")
    print(f"  {'Column':<22} {'Mean':>8} {'Median':>8} {'Mode':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 62)

    manual_stats = {}
    for col in stat_cols:
        vals = df[col].tolist()
        mn   = round(manual_mean(vals), 3)
        med  = round(manual_median(vals), 3)
        mod  = round(manual_mode(vals), 3) if manual_mode(vals) is not None else "N/A"
        mi   = round(manual_min(vals), 3)
        ma   = round(manual_max(vals), 3)
        manual_stats[col] = (mn, med, mod, mi, ma)
        print(f"  {col:<22} {mn:>8} {med:>8} {str(mod):>8} {mi:>8} {ma:>8}")

    print("\n--- Pandas Statistics ---")
    pd_desc = df[stat_cols].describe()
    print(pd_desc.to_string())

    print("\n--- Group Statistics by Outcome (Diabetic vs Non-Diabetic) ---")
    outcome_groups = {0: {col: [] for col in stat_cols},
                      1: {col: [] for col in stat_cols}}

    for i in range(len(df)):
        outcome = int(df.at[i, 'Outcome'])
        for col in stat_cols:
            outcome_groups[outcome][col].append(df.at[i, col])

    labels = {0: "Non-Diabetic (Outcome=0)", 1: "Diabetic (Outcome=1)"}
    for outcome_val, group_data in outcome_groups.items():
        print(f"\n  Group: {labels[outcome_val]} | Count: {len(group_data['Glucose'])}")
        print(f"    {'Column':<22} {'Mean':>9} {'Median':>9} {'Min':>9} {'Max':>9}")
        print("    " + "-" * 50)
        for col in stat_cols:
            vals = group_data[col]
            mn   = round(manual_mean(vals), 3)
            med  = round(manual_median(vals), 3)
            mi   = round(manual_min(vals), 3)
            ma   = round(manual_max(vals), 3)
            print(f"    {col:<22} {mn:>9} {med:>9} {mi:>9} {ma:>9}")

    print("\n--- Unique Values Using Sets ---")
    for col in ['Age', 'Pregnancies', 'Outcome']:
        unique_set = set(df[col].tolist())
        print(f"  Unique {col:<15}: {sorted(unique_set)}")

    print("\n--- Group Statistics by Age_Risk_Level ---")
    risk_level_data = {}
    for i in range(len(df)):
        level = df.at[i, 'Age_Risk_Level']
        if level not in risk_level_data:
            risk_level_data[level] = {'Glucose': [], 'BMI': [], 'Outcome': []}
        risk_level_data[level]['Glucose'].append(df.at[i, 'Glucose'])
        risk_level_data[level]['BMI'].append(df.at[i, 'BMI'])
        risk_level_data[level]['Outcome'].append(df.at[i, 'Outcome'])

    for level, data in risk_level_data.items():
        n_level      = len(data['Glucose'])
        avg_glucose  = round(manual_mean(data['Glucose']), 2)
        avg_bmi      = round(manual_mean(data['BMI']), 2)
        diabetes_pct = round(sum(data['Outcome']) / n_level * 100, 1)
        print(f"  {level:<15} | N={n_level:3d} | Avg Glucose={avg_glucose:6.2f} "
              f"| Avg BMI={avg_bmi:5.2f} | Diabetes%={diabetes_pct}%")

    return manual_stats, outcome_groups

def generate_report(df, manual_stats, outcome_groups):
    """
    Generates a detailed statistical analysis report and saves it as
    diabetes_analysis_report.txt using file write inside try-except.
    """
    print("\n" + "=" * 65)
    print("  OPERATION 9 : GENERATING FINAL REPORT")
    print("=" * 65)

    stat_cols = ['Glucose', 'BMI', 'Age', 'Insulin',
                 'BloodPressure', 'Pregnancies', 'BMI_Glucose_Index']
    labels    = {0: "Non-Diabetic", 1: "Diabetic"}

    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("   PIMA INDIANS DIABETES - STATISTICAL ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"   Student  : PATTELA HEMANTH")
    report_lines.append(f"   Reg No   : AP24110010671")
    report_lines.append(f"   Section  : E")
    report_lines.append(f"   Subject  : CSE 205 - Hands on with Python")
    report_lines.append(f"   Project  : 5 - Pima Indians Diabetes Statistical Analysis")
    report_lines.append("=" * 70)
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 1: DATASET OVERVIEW")
    report_lines.append("-" * 70)
    report_lines.append(f"  Dataset    : Pima Indians Diabetes Dataset")
    report_lines.append(f"  Source     : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    report_lines.append(f"  Total Rows : {len(df)}")
    report_lines.append(f"  Columns    : {df.shape[1]}")
    report_lines.append(f"  Features   : {', '.join(df.columns.tolist())}")
    report_lines.append(f"  Outcome 0  : {(df['Outcome'] == 0).sum()} (Non-Diabetic)")
    report_lines.append(f"  Outcome 1  : {(df['Outcome'] == 1).sum()} (Diabetic)")
    diabetes_pct = (df['Outcome'] == 1).sum() / len(df) * 100
    report_lines.append(f"  Diabetes Prevalence : {diabetes_pct:.2f}%")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 2: MISSING VALUE HANDLING")
    report_lines.append("-" * 70)
    report_lines.append("  Columns treated for zero-as-missing: Glucose, BloodPressure,")
    report_lines.append("  SkinThickness, Insulin, BMI.")
    report_lines.append("  Zeros replaced with manually computed column medians.")
    report_lines.append(f"  Remaining NaN values: {df.isnull().sum().sum()}")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 3: ENGINEERED FEATURES (NEW COLUMNS)")
    report_lines.append("-" * 70)
    report_lines.append("  i)  BMI_Glucose_Index     = (BMI * Glucose) / 100")
    report_lines.append("  ii) Insulin_Glucose_Ratio = Insulin / Glucose")
    report_lines.append("  iii)Age_Risk_Level         = Young / Middle-Aged / Senior")
    report_lines.append(f"  BMI_Glucose_Index   → Mean={df['BMI_Glucose_Index'].mean():.3f}, "
                        f"Min={df['BMI_Glucose_Index'].min():.3f}, Max={df['BMI_Glucose_Index'].max():.3f}")
    report_lines.append(f"  Insulin_Glucose_Ratio → Mean={df['Insulin_Glucose_Ratio'].mean():.4f}")
    age_dist = df['Age_Risk_Level'].value_counts().to_dict()
    report_lines.append(f"  Age_Risk_Level distribution: {age_dist}")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 4: HIGH-RISK FLAG ANALYSIS")
    report_lines.append("-" * 70)
    report_lines.append("  Condition: (Glucose > 140) OR (BMI > 30) OR (Age > 50)")
    hr_count    = df['High_Risk'].sum()
    nhr_count   = len(df) - hr_count
    hr_diabetic = ((df['High_Risk'] == 1) & (df['Outcome'] == 1)).sum()
    nhr_diabetic= ((df['High_Risk'] == 0) & (df['Outcome'] == 1)).sum()
    hr_rate     = hr_diabetic / hr_count * 100 if hr_count else 0
    nhr_rate    = nhr_diabetic / nhr_count * 100 if nhr_count else 0
    report_lines.append(f"  High-Risk Patients          : {hr_count} ({hr_count/len(df)*100:.1f}%)")
    report_lines.append(f"  Non-High-Risk Patients      : {nhr_count}")
    report_lines.append(f"  Diabetes Rate (High-Risk)   : {hr_rate:.1f}%")
    report_lines.append(f"  Diabetes Rate (Non-HR)      : {nhr_rate:.1f}%")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 5: MANUAL STATISTICS (loops + recursion)")
    report_lines.append("-" * 70)
    report_lines.append(f"  {'Column':<22} {'Mean':>9} {'Median':>9} {'Min':>9} {'Max':>9}")
    report_lines.append("  " + "-" * 60)
    for col, (mn, med, mod, mi, ma) in manual_stats.items():
        report_lines.append(f"  {col:<22} {mn:>9} {med:>9} {mi:>9} {ma:>9}")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 6: PANDAS STATISTICS")
    report_lines.append("-" * 70)
    pd_desc = df[stat_cols].describe()
    report_lines.append(pd_desc.to_string())
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 7: GROUP STATISTICS BY OUTCOME")
    report_lines.append("-" * 70)
    for outcome_val, group_data in outcome_groups.items():
        n_grp = len(group_data['Glucose'])
        report_lines.append(f"\n  [{labels[outcome_val]}] — {n_grp} patients")
        report_lines.append(f"  {'Column':<22} {'Mean':>9} {'Median':>9} {'Min':>9} {'Max':>9}")
        report_lines.append("  " + "-" * 55)
        for col in stat_cols:
            vals = group_data[col]
            mn   = round(manual_mean(vals), 3)
            med  = round(manual_median(vals), 3)
            mi   = round(manual_min(vals), 3)
            ma   = round(manual_max(vals), 3)
            report_lines.append(f"  {col:<22} {mn:>9} {med:>9} {mi:>9} {ma:>9}")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 8: GROUP STATISTICS BY AGE_RISK_LEVEL")
    report_lines.append("-" * 70)
    risk_level_data = {}
    for i in range(len(df)):
        level = df.at[i, 'Age_Risk_Level']
        if level not in risk_level_data:
            risk_level_data[level] = {'Glucose': [], 'BMI': [], 'Outcome': [], 'Insulin': []}
        risk_level_data[level]['Glucose'].append(df.at[i, 'Glucose'])
        risk_level_data[level]['BMI'].append(df.at[i, 'BMI'])
        risk_level_data[level]['Outcome'].append(df.at[i, 'Outcome'])
        risk_level_data[level]['Insulin'].append(df.at[i, 'Insulin'])

    for level in ['Young', 'Middle-Aged', 'Senior']:
        if level not in risk_level_data:
            continue
        data         = risk_level_data[level]
        n_level      = len(data['Glucose'])
        avg_glucose  = round(manual_mean(data['Glucose']), 2)
        avg_bmi      = round(manual_mean(data['BMI']), 2)
        avg_insulin  = round(manual_mean(data['Insulin']), 2)
        level_diabetes_pct = round(sum(data['Outcome']) / n_level * 100, 1)
        report_lines.append(f"\n  {level} (N={n_level}):")
        report_lines.append(f"    Avg Glucose    : {avg_glucose}")
        report_lines.append(f"    Avg BMI        : {avg_bmi}")
        report_lines.append(f"    Avg Insulin    : {avg_insulin}")
        report_lines.append(f"    Diabetes Rate  : {level_diabetes_pct}%")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 9: FUNCTIONAL PROGRAMMING (filter, map, reduce)")
    report_lines.append("-" * 70)
    insulin_list = df['Insulin'].tolist()
    glucose_list = df['Glucose'].tolist()
    pairs        = list(zip(insulin_list, glucose_list))
    filtered     = list(filter(lambda p: p[0] > 0, pairs))
    ig_ratios    = list(map(lambda p: round(p[0]/p[1], 4) if p[1] != 0 else 0.0, filtered))
    total_ratio  = reduce(lambda a, b: a + b, ig_ratios) if ig_ratios else 0
    avg_ratio    = total_ratio / len(ig_ratios) if ig_ratios else 0
    report_lines.append(f"  filter() → Rows with Insulin > 0   : {len(filtered)}")
    report_lines.append(f"  map()    → Insulin/Glucose ratios computed")
    report_lines.append(f"  reduce() → Sum of all ratios        : {total_ratio:.4f}")
    report_lines.append(f"             Average ratio            : {avg_ratio:.4f}")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 10: UNIQUE VALUES (Using Sets)")
    report_lines.append("-" * 70)
    unique_ages     = sorted(set(df['Age'].tolist()))
    unique_preg     = sorted(set(df['Pregnancies'].tolist()))
    unique_outcome  = sorted(set(df['Outcome'].tolist()))
    unique_risk_lvl = sorted(set(df['Age_Risk_Level'].tolist()))
    report_lines.append(f"  Unique Outcomes         : {unique_outcome}")
    report_lines.append(f"  Unique Age_Risk_Levels  : {unique_risk_lvl}")
    report_lines.append(f"  Unique Pregnancies      : {unique_preg}")
    report_lines.append(f"  Unique Age values       : {len(unique_ages)} distinct ages (range {min(unique_ages)}-{max(unique_ages)})")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("SECTION 11: KEY INSIGHTS & CONCLUSIONS")
    report_lines.append("-" * 70)
    avg_glucose_diabetic     = round(manual_mean(outcome_groups[1]['Glucose']), 2)
    avg_glucose_non_diabetic = round(manual_mean(outcome_groups[0]['Glucose']), 2)
    avg_bmi_diabetic         = round(manual_mean(outcome_groups[1]['BMI']), 2)
    avg_bmi_non_diabetic     = round(manual_mean(outcome_groups[0]['BMI']), 2)
    avg_age_diabetic         = round(manual_mean(outcome_groups[1]['Age']), 2)
    avg_age_non_diabetic     = round(manual_mean(outcome_groups[0]['Age']), 2)
    report_lines.append(f"  1. Diabetes Prevalence  : {diabetes_pct:.2f}% of the dataset")
    report_lines.append(f"  2. Glucose levels are significantly higher in diabetics:")
    report_lines.append(f"       Diabetic avg={avg_glucose_diabetic}  vs  Non-Diabetic avg={avg_glucose_non_diabetic}")
    report_lines.append(f"  3. BMI is higher in diabetic patients:")
    report_lines.append(f"       Diabetic avg={avg_bmi_diabetic}  vs  Non-Diabetic avg={avg_bmi_non_diabetic}")
    report_lines.append(f"  4. Diabetics tend to be older:")
    report_lines.append(f"       Diabetic avg age={avg_age_diabetic}  vs  Non-Diabetic avg age={avg_age_non_diabetic}")
    report_lines.append(f"  5. High-Risk patients have a diabetes rate of {hr_rate:.1f}% vs {nhr_rate:.1f}% in non-high-risk group.")
    report_lines.append(f"  6. {len(filtered)} patients had measurable insulin levels (Insulin > 0).")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("   END OF REPORT")
    report_lines.append("=" * 70)

    report_filename = "diabetes_analysis_report.txt"
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + "\n")
        print(f"\n  Report saved successfully: '{report_filename}'")
    except PermissionError:
        print(f"[ERROR] Permission denied: Cannot write '{report_filename}'.")
    except OSError as e:
        print(f"[ERROR] File write failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error while writing report: {e}")

    return report_filename

def main():
    print("\n" + "=" * 65)
    print("  CSE 205 - PIMA INDIANS DIABETES STATISTICAL ANALYSIS")
    print("  Student: PATTELA HEMANTH  |  Reg No: AP24110010671")
    print("=" * 65)

    df = load_dataset("diabetes.csv")

    explore_data(df)

    df = handle_missing_values(df)

    df = create_new_columns(df)

    df = high_risk_flags(df)

    insulin_glucose_ratios(df)

    recursive_total_risk(df)

    manual_stats, outcome_groups = statistics_calculation(df)

    report_file = generate_report(df, manual_stats, outcome_groups)

    print("\n" + "=" * 65)
    print("  ALL 9 OPERATIONS COMPLETED SUCCESSFULLY")
    print(f"  Report saved as: {report_file}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()