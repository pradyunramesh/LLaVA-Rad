import pandas as pd
import numpy as np
from logger import logger
from sklearn.metrics import f1_score
from itertools import combinations
from scipy import stats
from sklearn.metrics import confusion_matrix

original_columns = ['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
       'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']
labeled_columns = [col + '_Labeled' for col in original_columns]
race_columns = ['race_WHITE', 'race_BLACK', 'race_ASIAN', 'race_HISPANIC']
sex_columns = ['Female', 'Male']
age_columns = ['anchor_age_0.0','anchor_age_1.0','anchor_age_2.0','anchor_age_3.0','anchor_age_4.0']
race_sex_columns = ['race_WHITE_Female', 'race_WHITE_Male', 'race_BLACK_Female', 'race_BLACK_Male','race_ASIAN_Female', 'race_ASIAN_Male', 'race_HISPANIC_Female', 'race_HISPANIC_Male']
race_age_columns = ['race_WHITE_anchor_age_0.0', 'race_WHITE_anchor_age_1.0', 'race_WHITE_anchor_age_2.0', 'race_WHITE_anchor_age_3.0', 'race_WHITE_anchor_age_4.0',
                        'race_BLACK_anchor_age_0.0', 'race_BLACK_anchor_age_1.0', 'race_BLACK_anchor_age_2.0', 'race_BLACK_anchor_age_3.0', 'race_BLACK_anchor_age_4.0',
                        'race_ASIAN_anchor_age_0.0', 'race_ASIAN_anchor_age_1.0', 'race_ASIAN_anchor_age_2.0', 'race_ASIAN_anchor_age_3.0', 'race_ASIAN_anchor_age_4.0',
                        'race_HISPANIC_anchor_age_0.0', 'race_HISPANIC_anchor_age_1.0', 'race_HISPANIC_anchor_age_2.0', 'race_HISPANIC_anchor_age_3.0', 'race_HISPANIC_anchor_age_4.0']

def compute_tpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return tpr

def compute_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return fpr

def calculate_p_value(obs_value, metrics, score_type, x, y):
    if obs_value >= 0:
        p_value = np.sum(metrics <= 0) / len(metrics)
    else:
        p_value = np.sum(metrics >= 0) / len(metrics)
    significance = f"No statistical significance for {score_type} score"
    if p_value <0.05:
        if obs_value >= 0:
            significance = f"{x} has a significantly higher {score_type} score than {y}"
        else:
            significance = f"{y} has a significantly higher {score_type} score than {x}"
    return p_value, significance

def calculate_metric(df, metric, original_columns, labeled_columns):
    if metric == "loss":
        return df['loss'].mean()
    elif metric == "f1":
        return f1_score(original_columns, labeled_columns, average = 'binary')
    elif metric == "tpr":
        return compute_tpr(original_columns, labeled_columns)
    elif metric == "fpr":
        return compute_fpr(original_columns, labeled_columns)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def evaluate_race_condition_table(test_chexpert_reports, model_type, metric):
    # Calculate table for race vs condition
    if metric == "loss":
        labeled_columns = ['No Finding_Labeled']  # Dummy column to avoid empty list
    else:
        labeled_columns = [col + '_Labeled' for col in original_columns]
    race_condition_df = pd.DataFrame(columns = ['Condition', 'Race 1', 'Race 2', f'Race 1: {metric}', f'Race 2: {metric}', f'P-Value {metric}', 
    f'Statistical Significance {metric}'])
    for col in range(len(labeled_columns)):
        for x, y in combinations(race_columns, 2):
            #Calculate observed difference in metrics
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports_1[labeled_columns[col]]
            original_col = chexpert_reports_1[original_columns[col]]
            score_1_og = calculate_metric(chexpert_reports_1, metric, original_col, labeled_col)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports_2[labeled_columns[col]]
            original_col = chexpert_reports_2[original_columns[col]]
            score_2_og = calculate_metric(chexpert_reports_2, metric, original_col, labeled_col)

            obs_value = score_1_og - score_2_og

            # Perform bootstrap sampling to evaluate t-test with statistical significance
            scores = []
            for num in range(10000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                race1_score = calculate_metric(chexpert_reports_1_sample, metric, original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                race2_score = calculate_metric(chexpert_reports_2_sample, metric, original_col_2, labeled_col_2)

                scores.append(race1_score - race2_score)

            scores = np.array(scores)

            p_value, significance = calculate_p_value(obs_value, scores, metric, x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Race 1': x, 'Race 2': y, f'Race 1: {metric}': score_1_og, 
                f'Race 2: {metric}': score_2_og, f'P-Value {metric}': p_value, f'Statistical Significance {metric}': significance
            }])
            race_condition_df = pd.concat([race_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        race_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/race_condition_{metric}.csv')
    elif model_type == "finetuned":
        race_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/race_condition_{metric}.csv')

def evaluate_sex_condition_table(test_chexpert_reports, model_type, metric):
    ## Calculate table for sex vs condition
    if metric == "loss":
        labeled_columns = ['No Finding_Labeled']  # Dummy column to avoid empty list
    else:
        labeled_columns = [col + '_Labeled' for col in original_columns]
    sex_condition_df = pd.DataFrame(columns = ['Condition', 'Sex 1', 'Sex 2', f'Sex 1: {metric}', f'Sex 2: {metric}', f'P-Value {metric}', 
    f'Statistical Significance {metric}'])
    for col in range(len(labeled_columns)):
        # Calculate observed F1 score
        for x, y in combinations(sex_columns, 2):
            #Calculate observed difference in F1 scores
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == x].reset_index(drop = True)
            labeled_col_1 = chexpert_reports_1[labeled_columns[col]]
            original_col_1 = chexpert_reports_1[original_columns[col]]
            score_1_og = calculate_metric(chexpert_reports_1, metric, original_col_1, labeled_col_1)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == y].reset_index(drop = True)
            labeled_col_2 = chexpert_reports_2[labeled_columns[col]]
            original_col_2 = chexpert_reports_2[original_columns[col]]
            score_2_og = calculate_metric(chexpert_reports_2, metric, original_col_2, labeled_col_2)

            obs_value = score_1_og - score_2_og

            # Perform bootstrap sampling to evaluate t-test with statistical significance
            scores = []
            for num in range(10000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == x].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                score_1 = calculate_metric(chexpert_reports_1_sample, metric, original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == y].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                score_2 = calculate_metric(chexpert_reports_2_sample, metric, original_col_2, labeled_col_2)

                scores.append(score_1 - score_2)

            scores = np.array(scores)

            p_value, significance = calculate_p_value(obs_value, scores, metric, x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Sex 1': x, 'Sex 2': y, f'Sex 1: {metric}': score_1_og, 
                f'Sex 2: {metric}': score_2_og, f'P-Value {metric}': p_value, f'Statistical Significance {metric}': significance
            }])
            sex_condition_df = pd.concat([sex_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        sex_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/sex_condition_{metric}.csv')
    elif model_type == "finetuned":
        sex_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/sex_condition_{metric}.csv')

def evaluate_age_condition_table(test_chexpert_reports, model_type, metric):
    ## Calculate table for age vs condition
    if metric == "loss":
        labeled_columns = ['No Finding_Labeled']  # Dummy column to avoid empty list
    else:
        labeled_columns = [col + '_Labeled' for col in original_columns]
    age_condition_df = pd.DataFrame(columns = ['Condition', 'Age 1', 'Age 2', f'Age 1: {metric}', f'Age 2: {metric}', f'P-Value {metric}', 
    f'Statistical Significance {metric}'])
    for col in range(len(labeled_columns)):
        # Calculate observed F1 score
        for x, y in combinations(age_columns, 2):
            #Calculate observed difference in F1 scores
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports_1[labeled_columns[col]]
            original_col = chexpert_reports_1[original_columns[col]]
            score_1_og = calculate_metric(chexpert_reports_1, metric, original_col, labeled_col)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports_2[labeled_columns[col]]
            original_col = chexpert_reports_2[original_columns[col]]
            score_2_og = calculate_metric(chexpert_reports_2, metric, original_col, labeled_col)

            obs_value = score_1_og - score_2_og

            # Perform bootstrap sampling to evaluate t-test with statistical significance
            scores = []
            for num in range(10000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                age1_score = calculate_metric(chexpert_reports_1_sample, metric, original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                age2_score = calculate_metric(chexpert_reports_2_sample, metric, original_col_2, labeled_col_2)

                scores.append(age1_score - age2_score)

            scores = np.array(scores)
            p_value, significance = calculate_p_value(obs_value, scores, metric, x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Age 1': x, 'Age 2': y, f'Age 1: {metric}': score_1_og, 
                f'Age 2: {metric}': score_2_og, f'P-Value {metric}': p_value, f'Statistical Significance {metric}': significance
            }])
            age_condition_df = pd.concat([age_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        age_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/age_condition_{metric}.csv')
    elif model_type == "finetuned":
        age_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/age_condition_{metric}.csv')

def evaluate_race_sex_condition_table(test_chexpert_reports, model_type, metric):
    ## Calculate table for race vs sex vs condition
    if metric == "loss":
        labeled_columns = ['No Finding_Labeled']  # Dummy column to avoid empty list
    else:
        labeled_columns = [col + '_Labeled' for col in original_columns]
    race_sex_condition_df = pd.DataFrame(columns=[
        'Condition', 'Group 1', 'Group 2',
        f'Group 1: {metric}', f'Group 2: {metric}', f'P-Value {metric}', f'Statistical Significance {metric}',
    ])
    for col in range(len(labeled_columns)):
        for x, y in combinations(race_sex_columns, 2):
            x_parts = x.rsplit('_', 1)
            race_1 = x_parts[0]
            sex_1 = x_parts[1]
            chexpert_reports = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
            chexpert_reports = chexpert_reports[chexpert_reports['sex'] == sex_1].reset_index(drop = True)
            labeled_col = chexpert_reports[labeled_columns[col]]
            original_col = chexpert_reports[original_columns[col]]
            score_1_og = calculate_metric(chexpert_reports, metric, original_col, labeled_col)

            y_parts = y.rsplit('_', 1)
            race_2 = y_parts[0]
            sex_2 = y_parts[1]
            chexpert_reports = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
            chexpert_reports = chexpert_reports[chexpert_reports['sex'] == sex_2].reset_index(drop = True)
            labeled_col = chexpert_reports[labeled_columns[col]]
            original_col = chexpert_reports[original_columns[col]]
            score_2_og = calculate_metric(chexpert_reports, metric, original_col, labeled_col)

            obs_value = score_1_og - score_2_og

            scores = []
            for num in range(10000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
                chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1['sex'] == sex_1].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                group1_score = calculate_metric(chexpert_reports_1_sample, metric, original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
                chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2['sex'] == sex_2].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                group2_score = calculate_metric(chexpert_reports_2_sample, metric, original_col_2, labeled_col_2)

                scores.append(group1_score - group2_score)

            scores = np.array(scores)
            p_value, significance = calculate_p_value(obs_value, scores, metric, x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Group 1': x, 'Group 2': y, f'Group 1: {metric}': score_1_og, 
                f'Group 2: {metric}': score_2_og, f'P-Value {metric}': p_value, f'Statistical Significance {metric}': significance,
            }])
            race_sex_condition_df = pd.concat([race_sex_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        race_sex_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/race_sex_condition_{metric}.csv')
    elif model_type == "finetuned":
        race_sex_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/race_sex_condition_{metric}.csv')

def evaluate_race_age_condition_table(test_chexpert_reports, model_type, metric):
    ## Calculate table for race vs age vs condition
    if metric == "loss":
        labeled_columns = ['No Finding_Labeled']  # Dummy column to avoid empty list
    else:
        labeled_columns = [col + '_Labeled' for col in original_columns]
    race_age_condition_df = pd.DataFrame(columns=[
        'Condition', 'Group 1', 'Group 2',
        f'Group 1: {metric}', f'Group 2: {metric}', f'P-Value {metric}', f'Statistical Significance {metric}'
    ])
    for col in range(len(labeled_columns)):
        for x, y in combinations(race_age_columns, 2):
            x_parts = x.split('_')
            race_1 = '_'.join(x_parts[:2])
            age_1 = '_'.join(x_parts[2:])
            chexpert_reports = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
            chexpert_reports = chexpert_reports[chexpert_reports[age_1] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports[labeled_columns[col]]
            original_col = chexpert_reports[original_columns[col]]
            score_1_og = calculate_metric(chexpert_reports, metric, original_col, labeled_col)

            y_parts = y.split('_')
            race_2 = '_'.join(y_parts[:2])
            age_2 = '_'.join(y_parts[2:])
            chexpert_reports = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
            chexpert_reports = chexpert_reports[chexpert_reports[age_2] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports[labeled_columns[col]]
            original_col = chexpert_reports[original_columns[col]]
            score_2_og = calculate_metric(chexpert_reports, metric, original_col, labeled_col)

            obs_value = score_1_og - score_2_og

            scores = []
            for num in range(10000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
                chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1[age_1] == 1].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                group1_score = calculate_metric(chexpert_reports_1_sample, metric, original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
                chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2[age_2] == 1].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                group2_score = calculate_metric(chexpert_reports_2_sample, metric, original_col_2, labeled_col_2)

                scores.append(group1_score - group2_score)

            scores = np.array(scores)

            p_value, significance = calculate_p_value(obs_value, scores, metric, x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Group 1': x, 'Group 2': y, f'Group 1: {metric}': score_1_og, 
                f'Group 2: {metric}': score_2_og, f'P-Value {metric}': p_value, f'Statistical Significance {metric}': significance
            }])
            race_age_condition_df = pd.concat([race_age_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        race_age_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/race_age_condition_{metric}.csv')
    elif model_type == "finetuned":
        race_age_condition_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/race_age_condition_{metric}.csv')
