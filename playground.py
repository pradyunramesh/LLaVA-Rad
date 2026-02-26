import pandas as pd
import numpy as np
import os

df = pd.read_csv("/home/pr2762@mc.cumc.columbia.edu/condition_attributions_data.csv")

def calculate_accuracy(y_pred, y_true):
    '''
    Method to calculate accuracy given predicted and true labels
    '''
    y_pred = y_pred.to_numpy()
    y_true = y_true.to_numpy()
    label_accuracies = (y_pred == y_true).mean(axis=0)
    average_accuracy = label_accuracies.mean()
    return average_accuracy

def single_accuracy(data, subset_cols = None, weight = None, target_name = None):
    if subset_cols is None:
        raise ValueError("Subset columns are required")
    if target_name is None:
        raise ValueError("Target columns are required")
    preds = data[subset_cols].values.astype(int)
    labels = data[target_name].values.astype(int)

    if preds.ndim != 1 or labels.ndim != 1:
        raise ValueError("Single-label accuracy requires 1D preds and labels")

    correct = preds == labels  # shape (N,)

    if weight is None:
        return correct.mean()
    else:
        weight = np.asarray(weight).reshape(-1)
        if weight.shape[0] != preds.shape[0]:
            raise ValueError("Weight must have shape (N,)")
        return np.sum(weight * correct) / np.sum(weight)

#From an accuracy standpoint, the subset_cols should be the labeled_columns and the target_name should be the original columns
def multi_accuracy(data, subset_cols = None, weight = None, target_name = None):

    if subset_cols is None:
        raise ValueError("Subset columns are required")
    if target_name is None:
        raise ValueError("Target columns are required")
    preds = data[subset_cols].values.astype(int)
    labels = data[target_name].values.astype(int)

    if weight is None:
        weight = np.ones(preds.shape[0])
    else:
        weight = weight.reshape(-1) #The dimensions of this weight should be (N,)
    
    weight = weight[:, None] #Weights are now (N,1)
    correct = preds == labels #Boolean (N,C) array
    weighted_correct = weight * correct #Numpy copies the single column in weight across C columns
    weighted_total = weight * np.ones_like(correct)
    return weighted_correct.sum() / weighted_total.sum()

def calculate_tpr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    if TP + FN == 0:
        return 0.0  # no positive labels → define TPR as 0
    return TP / (TP + FN)

def calculate_fpr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    if FP + TN == 0:
        return 0.0  # no negatives → define FPR as 0
    return FP / (FP + TN)

def calculate_f1(y_true, y_pred):
    """Calculate F1 score (harmonic mean of precision and recall)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    # Recall = TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

race_columns = ['race_WHITE', 'race_BLACK', 'race_ASIAN', 'race_HISPANIC', 'race_Other', 'race_UNKNOWN']
age_columns = ['anchor_age_0.0','anchor_age_1.0','anchor_age_2.0','anchor_age_3.0','anchor_age_4.0']
original_columns = ['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
       'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']
labeled_columns = [col + '_Labeled' for col in original_columns]

df["race"] = df[race_columns].idxmax(axis=1)
df["age"] = df[age_columns].idxmax(axis=1)
df["demo_group"] = df["race"] + "_" + df["age"] + "_" + df["sex"]

# ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Extract predictions and true labels
y_pred = df[labeled_columns]
y_true = df[original_columns]

# Calculate overall multi-label accuracy
overall_accuracy = calculate_accuracy(y_pred, y_true)

# Calculate accuracy by race
accuracy_by_race = []
for race in df["race"].unique():
    mask = df["race"] == race
    acc = calculate_accuracy(y_pred[mask], y_true[mask])
    accuracy_by_race.append({"race": race, "accuracy": acc, "count": mask.sum()})
race_df = pd.DataFrame(accuracy_by_race)
race_df.to_csv("outputs/accuracy_by_race.csv", index=False)

# Calculate accuracy by age
accuracy_by_age = []
for age in df["age"].unique():
    mask = df["age"] == age
    acc = calculate_accuracy(y_pred[mask], y_true[mask])
    accuracy_by_age.append({"age": age, "accuracy": acc, "count": mask.sum()})
age_df = pd.DataFrame(accuracy_by_age)
age_df.to_csv("outputs/accuracy_by_age.csv", index=False)

# Calculate accuracy by sex
accuracy_by_sex = []
for sex in df["sex"].unique():
    mask = df["sex"] == sex
    acc = calculate_accuracy(y_pred[mask], y_true[mask])
    accuracy_by_sex.append({"sex": sex, "accuracy": acc, "count": mask.sum()})
sex_df = pd.DataFrame(accuracy_by_sex)
sex_df.to_csv("outputs/accuracy_by_sex.csv", index=False)

# Calculate accuracy by demographic group
accuracy_by_demo = []
for demo_group in df["demo_group"].unique():
    mask = df["demo_group"] == demo_group
    acc = calculate_accuracy(y_pred[mask], y_true[mask])
    accuracy_by_demo.append({"demographic_group": demo_group, "accuracy": acc, "count": mask.sum()})
demo_df = pd.DataFrame(accuracy_by_demo)
demo_df.to_csv("outputs/accuracy_by_demographic_group.csv", index=False)

print(f"Overall Multi-label Accuracy: {overall_accuracy:.4f}")
print(f"Files saved to outputs/: accuracy_by_race.csv, accuracy_by_age.csv, accuracy_by_sex.csv, accuracy_by_demographic_group.csv")

# New helper: compute single-label accuracy, TPR, FPR for binary labels
def calculate_binary_metrics(y_true_single, y_pred_single):
    y_true_arr = np.array(y_true_single)
    y_pred_arr = np.array(y_pred_single)
    acc = float((y_true_arr == y_pred_arr).mean()) if y_true_arr.size > 0 else np.nan
    tpr = calculate_tpr(y_true_arr, y_pred_arr)
    fpr = calculate_fpr(y_true_arr, y_pred_arr)
    return acc, tpr, fpr

def per_condition_metrics_by_group(df, condition, true_col, pred_col, group_col):
    rows = []
    for g in df[group_col].unique():
        mask = df[group_col] == g
        count = int(mask.sum())
        if count == 0:
            acc = tpr = fpr = np.nan
        else:
            acc, tpr, fpr = calculate_binary_metrics(df.loc[mask, true_col], df.loc[mask, pred_col])
        rows.append({
            'condition': condition,
            group_col: g,
            'accuracy': acc,
            'tpr': tpr,
            'fpr': fpr,
            'count': count
        })
    return pd.DataFrame(rows)

def per_condition_single_accuracy_by_group(df, condition, true_col, pred_col, group_col):
    """Compute single-label accuracy using single_accuracy function (no weights) grouped by demographic variable."""
    rows = []
    for g in df[group_col].unique():
        mask = df[group_col] == g
        count = int(mask.sum())
        if count == 0:
            acc = np.nan
        else:
            df_subset = df.loc[mask].copy()
            acc = single_accuracy(df_subset, subset_cols=pred_col, weight=None, target_name=true_col)
        rows.append({
            'condition': condition,
            group_col: g,
            'accuracy': acc,
            'count': count
        })
    return pd.DataFrame(rows)

def overall_multi_accuracy_by_group(df, pred_cols, true_cols, group_col):
    """Compute overall multi-label accuracy using multi_accuracy function (no weights) grouped by demographic variable."""
    rows = []
    for g in df[group_col].unique():
        mask = df[group_col] == g
        count = int(mask.sum())
        if count == 0:
            acc = np.nan
        else:
            df_subset = df.loc[mask].copy()
            acc = multi_accuracy(df_subset, subset_cols=pred_cols, weight=None, target_name=true_cols)
        rows.append({
            group_col: g,
            'accuracy': acc,
            'count': count
        })
    return pd.DataFrame(rows)

def per_condition_f1_by_group(df, condition, true_col, pred_col, group_col):
    """Compute per-condition F1 score grouped by demographic variable."""
    rows = []
    for g in df[group_col].unique():
        mask = df[group_col] == g
        count = int(mask.sum())
        if count == 0:
            f1 = np.nan
        else:
            f1 = calculate_f1(df.loc[mask, true_col], df.loc[mask, pred_col])
        rows.append({
            'condition': condition,
            group_col: g,
            'f1': f1,
            'count': count
        })
    return pd.DataFrame(rows)

def overall_f1_by_group(df, pred_cols, true_cols, group_col):
    """Compute overall F1 score (macro-averaged across conditions) grouped by demographic variable."""
    rows = []
    for g in df[group_col].unique():
        mask = df[group_col] == g
        count = int(mask.sum())
        if count == 0:
            macro_f1 = np.nan
        else:
            df_subset = df.loc[mask]
            # Compute F1 for each condition and macro-average
            f1_scores = []
            for pred_col, true_col in zip(pred_cols, true_cols):
                f1 = calculate_f1(df_subset[true_col], df_subset[pred_col])
                f1_scores.append(f1)
            macro_f1 = np.mean(f1_scores)
        rows.append({
            group_col: g,
            'f1': macro_f1,
            'count': count
        })
    return pd.DataFrame(rows)

def condition_label_counts_by_group(df, condition, true_col, group_col):
    """Count positive and negative labels for a condition grouped by demographic variable."""
    rows = []
    for g in df[group_col].unique():
        mask = df[group_col] == g
        df_subset = df.loc[mask]
        positive_count = int((df_subset[true_col] == 1).sum())
        negative_count = int((df_subset[true_col] == 0).sum())
        total_count = int(mask.sum())
        positive_ratio = positive_count / total_count if total_count > 0 else 0.0
        negative_ratio = negative_count / total_count if total_count > 0 else 0.0
        rows.append({
            'condition': condition,
            group_col: g,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_count': total_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio
        })
    return pd.DataFrame(rows)

# Compute per-condition metrics grouped by race and by demographic group
cond_rows_race = []
cond_rows_demo = []
cond_rows_sex = []
cond_rows_age = []
for cond in original_columns:
    # ground truth is in the original column name; predictions are in the labeled columns
    true_col = cond
    pred_col = (cond + '_Labeled')
    # by race
    df_cond_race = per_condition_metrics_by_group(df, cond, true_col, pred_col, 'race')
    cond_rows_race.append(df_cond_race)
    # by demographic group
    df_cond_demo = per_condition_metrics_by_group(df, cond, true_col, pred_col, 'demo_group')
    cond_rows_demo.append(df_cond_demo)
    # by sex
    df_cond_sex = per_condition_metrics_by_group(df, cond, true_col, pred_col, 'sex')
    cond_rows_sex.append(df_cond_sex)
    # by age
    df_cond_age = per_condition_metrics_by_group(df, cond, true_col, pred_col, 'age')
    cond_rows_age.append(df_cond_age)

cond_race_df = pd.concat(cond_rows_race, ignore_index=True)
cond_demo_df = pd.concat(cond_rows_demo, ignore_index=True)
cond_sex_df = pd.concat(cond_rows_sex, ignore_index=True)
cond_age_df = pd.concat(cond_rows_age, ignore_index=True)
cond_race_df.to_csv('outputs/condition_metrics_by_race.csv', index=False)
cond_demo_df.to_csv('outputs/condition_metrics_by_demographic_group.csv', index=False)
cond_sex_df.to_csv('outputs/condition_metrics_by_sex.csv', index=False)
cond_age_df.to_csv('outputs/condition_metrics_by_age.csv', index=False)

print('Per-condition metric files saved: outputs/condition_metrics_by_race.csv, outputs/condition_metrics_by_demographic_group.csv, outputs/condition_metrics_by_sex.csv, outputs/condition_metrics_by_age.csv')

# Compute single_accuracy based metrics for comparison
print("\n=== Computing single_accuracy based metrics for comparison ===")
single_acc_rows_race = []
single_acc_rows_demo = []
for cond in original_columns:
    true_col = cond
    pred_col = (cond + '_Labeled')
    # by race
    df_single_race = per_condition_single_accuracy_by_group(df, cond, true_col, pred_col, 'race')
    single_acc_rows_race.append(df_single_race)
    # by demographic group
    df_single_demo = per_condition_single_accuracy_by_group(df, cond, true_col, pred_col, 'demo_group')
    single_acc_rows_demo.append(df_single_demo)

single_acc_race_df = pd.concat(single_acc_rows_race, ignore_index=True)
single_acc_demo_df = pd.concat(single_acc_rows_demo, ignore_index=True)
single_acc_race_df.to_csv('outputs/condition_single_accuracy_by_race.csv', index=False)
single_acc_demo_df.to_csv('outputs/condition_single_accuracy_by_demographic_group.csv', index=False)

print('Single_accuracy based files saved: outputs/condition_single_accuracy_by_race.csv, outputs/condition_single_accuracy_by_demographic_group.csv')

# Compare accuracy values for first condition and first group as sanity check
first_cond = original_columns[0]
first_race = df['race'].unique()[0]

# From binary_metrics approach
binary_acc = cond_race_df[(cond_race_df['condition'] == first_cond) & (cond_race_df['race'] == first_race)]['accuracy'].values[0]

# From single_accuracy approach
single_acc = single_acc_race_df[(single_acc_race_df['condition'] == first_cond) & (single_acc_race_df['race'] == first_race)]['accuracy'].values[0]

print(f"\nComparison for {first_cond} / {first_race}:")
print(f"  Binary metrics accuracy: {binary_acc:.6f}")
print(f"  Single_accuracy result:  {single_acc:.6f}")
print(f"  Match: {np.isclose(binary_acc, single_acc)}")

# Compute multi_accuracy based metrics for overall accuracy comparison
print("\n=== Computing multi_accuracy based metrics for comparison ===")
multi_acc_race_df = overall_multi_accuracy_by_group(df, labeled_columns, original_columns, 'race')
multi_acc_age_df = overall_multi_accuracy_by_group(df, labeled_columns, original_columns, 'age')
multi_acc_sex_df = overall_multi_accuracy_by_group(df, labeled_columns, original_columns, 'sex')
multi_acc_demo_df = overall_multi_accuracy_by_group(df, labeled_columns, original_columns, 'demo_group')

multi_acc_race_df.to_csv('outputs/multi_accuracy_by_race.csv', index=False)
multi_acc_age_df.to_csv('outputs/multi_accuracy_by_age.csv', index=False)
multi_acc_sex_df.to_csv('outputs/multi_accuracy_by_sex.csv', index=False)
multi_acc_demo_df.to_csv('outputs/multi_accuracy_by_demographic_group.csv', index=False)

print('Multi_accuracy based files saved: outputs/multi_accuracy_by_*.csv')

# Compare calculate_accuracy vs multi_accuracy results
print("\n=== Comparison of calculate_accuracy vs multi_accuracy ===")

# Race comparison
race_comparison = race_df.rename(columns={'accuracy': 'calculate_acc'}).merge(
    multi_acc_race_df.rename(columns={'accuracy': 'multi_acc'}),
    on='race'
)
race_comparison['difference'] = (race_comparison['calculate_acc'] - race_comparison['multi_acc']).abs()
print("\nBy Race:")
print(race_comparison[['race', 'calculate_acc', 'multi_acc', 'difference']].to_string(index=False))
print(f"Max difference: {race_comparison['difference'].max():.10f}")

# Age comparison
age_comparison = age_df.rename(columns={'accuracy': 'calculate_acc'}).merge(
    multi_acc_age_df.rename(columns={'accuracy': 'multi_acc'}),
    on='age'
)
age_comparison['difference'] = (age_comparison['calculate_acc'] - age_comparison['multi_acc']).abs()
print("\nBy Age:")
print(age_comparison[['age', 'calculate_acc', 'multi_acc', 'difference']].to_string(index=False))
print(f"Max difference: {age_comparison['difference'].max():.10f}")

# Sex comparison
sex_comparison = sex_df.rename(columns={'accuracy': 'calculate_acc'}).merge(
    multi_acc_sex_df.rename(columns={'accuracy': 'multi_acc'}),
    on='sex'
)
sex_comparison['difference'] = (sex_comparison['calculate_acc'] - sex_comparison['multi_acc']).abs()
print("\nBy Sex:")
print(sex_comparison[['sex', 'calculate_acc', 'multi_acc', 'difference']].to_string(index=False))
print(f"Max difference: {sex_comparison['difference'].max():.10f}")

# Demographic group comparison (just first 10 rows)
demo_comparison = demo_df.rename(columns={'demographic_group': 'demo_group', 'accuracy': 'calculate_acc'}).merge(
    multi_acc_demo_df.rename(columns={'accuracy': 'multi_acc'}),
    on='demo_group'
)
demo_comparison['difference'] = (demo_comparison['calculate_acc'] - demo_comparison['multi_acc']).abs()
print("\nBy Demographic Group (first 10 rows):")
print(demo_comparison[['demo_group', 'calculate_acc', 'multi_acc', 'difference']].head(10).to_string(index=False))
print(f"Max difference: {demo_comparison['difference'].max():.10f}")
print(f"All values match within floating point precision: {(demo_comparison['difference'] < 1e-10).all()}")

# Compute F1 scores
print("\n=== Computing F1 scores ===")

# Per-condition F1 by race and demographic group
print("\nComputing per-condition F1 scores...")
f1_rows_race = []
f1_rows_demo = []
for cond in original_columns:
    true_col = cond
    pred_col = (cond + '_Labeled')
    # by race
    df_f1_race = per_condition_f1_by_group(df, cond, true_col, pred_col, 'race')
    f1_rows_race.append(df_f1_race)
    # by demographic group
    df_f1_demo = per_condition_f1_by_group(df, cond, true_col, pred_col, 'demo_group')
    f1_rows_demo.append(df_f1_demo)

f1_race_df = pd.concat(f1_rows_race, ignore_index=True)
f1_demo_df = pd.concat(f1_rows_demo, ignore_index=True)
f1_race_df.to_csv('outputs/condition_f1_by_race.csv', index=False)
f1_demo_df.to_csv('outputs/condition_f1_by_demographic_group.csv', index=False)

print('Per-condition F1 files saved: outputs/condition_f1_by_race.csv, outputs/condition_f1_by_demographic_group.csv')

# Overall F1 (macro-averaged across conditions) by race and demographic group
print("\nComputing overall F1 scores...")
overall_f1_race_df = overall_f1_by_group(df, labeled_columns, original_columns, 'race')
overall_f1_demo_df = overall_f1_by_group(df, labeled_columns, original_columns, 'demo_group')

overall_f1_race_df.to_csv('outputs/overall_f1_by_race.csv', index=False)
overall_f1_demo_df.to_csv('outputs/overall_f1_by_demographic_group.csv', index=False)

print('Overall F1 files saved: outputs/overall_f1_by_race.csv, outputs/overall_f1_by_demographic_group.csv')

# Display sample F1 results
print("\n=== Sample F1 Score Results ===")
print("\nPer-condition F1 by Race (first 10 rows):")
print(f1_race_df.head(10)[['condition', 'race', 'f1']].to_string(index=False))

print("\nOverall F1 (macro-averaged) by Race:")
print(overall_f1_race_df[['race', 'f1', 'count']].to_string(index=False))

print("\nOverall F1 (macro-averaged) by Demographic Group (first 10 rows):")
print(overall_f1_demo_df[['demo_group', 'f1', 'count']].head(10).to_string(index=False))

# Compute condition label counts
print("\n=== Computing condition label counts by demographic group ===")
label_counts_rows_race = []
label_counts_rows_demo = []
for cond in original_columns:
    true_col = cond
    # by race
    df_counts_race = condition_label_counts_by_group(df, cond, true_col, 'race')
    label_counts_rows_race.append(df_counts_race)
    # by demographic group
    df_counts_demo = condition_label_counts_by_group(df, cond, true_col, 'demo_group')
    label_counts_rows_demo.append(df_counts_demo)

label_counts_race_df = pd.concat(label_counts_rows_race, ignore_index=True)
label_counts_demo_df = pd.concat(label_counts_rows_demo, ignore_index=True)
label_counts_race_df.to_csv('outputs/condition_label_counts_by_race.csv', index=False)
label_counts_demo_df.to_csv('outputs/condition_label_counts_by_demographic_group.csv', index=False)

print('Label count files saved: outputs/condition_label_counts_by_race.csv, outputs/condition_label_counts_by_demographic_group.csv')

# Display sample label count results
print("\n=== Sample Label Count Results ===")
print("\nCondition label counts by Race (first 15 rows):")
print(label_counts_race_df.head(15)[['condition', 'race', 'positive_count', 'negative_count', 'total_count', 'positive_ratio', 'negative_ratio']].to_string(index=False))

print("\nCondition label counts by Demographic Group (first 10 rows):")
print(label_counts_demo_df.head(10)[['condition', 'demo_group', 'positive_count', 'negative_count', 'total_count', 'positive_ratio', 'negative_ratio']].to_string(index=False))
