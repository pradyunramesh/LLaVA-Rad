import pandas as pd
import numpy as np
from logger import logger
from itertools import combinations
from sklearn.metrics import f1_score
from scipy.stats import mannwhitneyu

# Implementation here is a non-parametric, one-tailed bootstrap test

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
sex_age_columns = ['Female_anchor_age_0.0', 'Female_anchor_age_1.0', 'Female_anchor_age_2.0', 'Female_anchor_age_3.0', 'Female_anchor_age_4.0',
                   'Male_anchor_age_0.0', 'Male_anchor_age_1.0', 'Male_anchor_age_2.0', 'Male_anchor_age_3.0', 'Male_anchor_age_4.0']

def calculate_accuracy(y_pred, y_true):
    '''
    Method to calculate accuracy given predicted and true labels
    '''
    y_pred = y_pred.to_numpy()
    y_true = y_true.to_numpy()
    label_accuracies = (y_pred == y_true).mean(axis=0)
    average_accuracy = label_accuracies.mean()
    return average_accuracy

def calculate_metric(y_pred, y_true, metric):
    '''
    Method to calculate a given metric
    '''
    if metric == "mann-whitney-accuracy":
        return calculate_accuracy(y_pred, y_true)
    elif metric == "mann-whitney-micro-F1":
        return f1_score(y_true, y_pred, average='micro')
    elif metric == "mann-whitney-macro-F1":
        return f1_score(y_true, y_pred, average='macro')
    else:
        raise ValueError(f'Unsupported metric: {metric}')

def evaluate_race_table_mann_test(test_chexpert_reports, model_type, metric):
    '''
    Method to evaluate the overall accuracy and make race-based comparisons
    '''
    logger.info(f'Evaluating race-based accuracy comparisons for {metric}')
    race_df = pd.DataFrame(columns = ['Race 1', 'Race 2', f'{metric}: Race 1', f'{metric}: Race 2', 'Observed Difference', 'P-Value', 'Statistical Significance', 'Direction'])
    for x, y in combinations(race_columns, 2):
        race1_scores = []
        race2_scores = []
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
        for num in range(10000):
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            race1_score = calculate_metric(labeled_col_1, original_col_1, metric)
            race1_scores.append(race1_score)

            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            race2_score = calculate_metric(labeled_col_2, original_col_2, metric)
            race2_scores.append(race2_score)

        race1_scores = np.array(race1_scores)
        race2_scores = np.array(race2_scores)
        stat, p_value = mannwhitneyu(race1_scores, race2_scores, alternative='two-sided')
        obs_diff = race1_scores.mean() - race2_scores.mean()
        direction = f"{x} > {y}" if obs_diff > 0 else f"{y} > {x}"

        new_data = pd.DataFrame([{'Race 1': x, 'Race 2': y, f'{metric}: Race 1': race1_scores.mean(), f'{metric}: Race 2': race2_scores.mean(), 'Observed Difference': obs_diff,'P-Value': p_value, 'Statistical Significance': 'Significant' if p_value < 0.05 else 'Not Significant', 'Direction': direction}])
        race_df = pd.concat([race_df, new_data], ignore_index=True)
    if model_type == "base":
        race_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/{metric}_scores/race_{metric}.csv')
    elif model_type == "finetuned":
        race_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/{metric}_scores/race_{metric}.csv')

def evaluate_sex_table_mann_test(test_chexpert_reports, model_type, metric):
    '''
    Method to evaluate the overall accuracy and make sex-based comparisons
    '''
    logger.info(f'Evaluating sex-based comparisons for {metric}')
    sex_df = pd.DataFrame(columns = ['Sex 1', 'Sex 2', f'{metric}: Sex 1', f'{metric}: Sex 2', 'Observed Difference', 'P-Value', 'Statistical Significance', 'Direction'])
    for x, y in combinations(sex_columns, 2):
        # Using 'sex' column for filtering as shown in condition_utils.py
        sex1_scores = []
        sex2_scores = []
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == x].reset_index(drop = True)
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == y].reset_index(drop = True)
        for num in range(10000):
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            sex1_score = calculate_metric(labeled_col_1, original_col_1, metric)
            sex1_scores.append(sex1_score)

            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            sex2_score = calculate_metric(labeled_col_2, original_col_2, metric)
            sex2_scores.append(sex2_score)

        sex1_scores = np.array(sex1_scores)
        sex2_scores = np.array(sex2_scores)
        stat, p_value = mannwhitneyu(sex1_scores, sex2_scores, alternative='two-sided')

        obs_diff = sex1_scores.mean() - sex2_scores.mean()
        direction = f"{x} > {y}" if obs_diff > 0 else f"{y} > {x}"
        new_data = pd.DataFrame([{'Sex 1': x, 'Sex 2': y, f'{metric}: Sex 1': sex1_scores.mean(), f'{metric}: Sex 2': sex2_scores.mean(), 'Observed Difference': obs_diff, 'P-Value': p_value, 'Statistical Significance': 'Significant' if p_value < 0.05 else 'Not Significant', 'Direction': direction}])
        sex_df = pd.concat([sex_df, new_data], ignore_index=True)
    if model_type == "base":
        sex_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/{metric}_scores/sex_{metric}.csv')
    elif model_type == "finetuned":
        sex_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/{metric}_scores/sex_{metric}.csv')

def evaluate_age_table_mann_test(test_chexpert_reports, model_type, metric):
    '''
    Method to evaluate the overall accuracy and make age-based comparisons
    '''
    logger.info(f"Evaluating age-based comparisons for {metric}")
    age_df = pd.DataFrame(columns = ['Age 1', 'Age 2', f'{metric}: Age 1', f'{metric}: Age 2', 'Observed Difference', 'P-Value', 'Statistical Significance', 'Direction'])
    for x, y in combinations(age_columns, 2):
        age1_scores = []
        age2_scores = []
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
        for num in range(10000):
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            age1_score = calculate_metric(labeled_col_1, original_col_1, metric)
            age1_scores.append(age1_score)

            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            age2_score = calculate_metric(labeled_col_2, original_col_2, metric)
            age2_scores.append(age2_score)

        age1_scores = np.array(age1_scores)
        age2_scores = np.array(age2_scores)
        stat, p_value = mannwhitneyu(age1_scores, age2_scores, alternative='two-sided')

        obs_diff = age1_scores.mean() - age2_scores.mean()
        direction = f"{x} > {y}" if obs_diff > 0 else f"{y} > {x}"
        new_data = pd.DataFrame([{'Age 1': x, 'Age 2': y, f'{metric}: Age 1': age1_scores.mean(), f'{metric}: Age 2': age2_scores.mean(), 'Observed Difference': obs_diff, 'P-Value': p_value, 'Statistical Significance': 'Significant' if p_value < 0.05 else 'Not Significant', 'Direction': direction}])
        age_df = pd.concat([age_df, new_data], ignore_index=True)
    if model_type == "base":
        age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/{metric}_scores/age_{metric}.csv')
    elif model_type == "finetuned":
        age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/{metric}_scores/age_{metric}.csv')

def evaluate_race_sex_table_mann_test(test_chexpert_reports, model_type, metric):
    '''
    Method to evaluate the overall accuracy and make intersectional race-sex comparisons
    '''
    logger.info(f"Evaluating intersectional race-sex comparisons for {metric}")
    race_sex_df = pd.DataFrame(columns = ['Group 1', 'Group 2', f'{metric}: Group 1', f'{metric}: Group 2', 'Observed Difference', 'P-Value', 'Statistical Significance', 'Direction'])
    for x, y in combinations(race_sex_columns, 2):
        group1_scores = []
        group2_scores = []
        x_parts = x.rsplit('_', 1)
        race_1 = x_parts[0]
        sex_1 = x_parts[1]
        y_parts = y.rsplit('_', 1)
        race_2 = y_parts[0]
        sex_2 = y_parts[1]
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
        chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1['sex'] == sex_1].reset_index(drop = True)
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
        chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2['sex'] == sex_2].reset_index(drop = True)
        for num in range(10000):
            # Bootstrap sampling for group 1
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            group1_score = calculate_metric(labeled_col_1, original_col_1, metric)
            group1_scores.append(group1_score)

            # Bootstrap sampling for group 2
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            group2_score = calculate_metric(labeled_col_2, original_col_2, metric)
            group2_scores.append(group2_score)

        group1_scores = np.array(group1_scores)
        group2_scores = np.array(group2_scores)
        stat, p_value = mannwhitneyu(group1_scores, group2_scores, alternative='two-sided')

        obs_diff = group1_scores.mean() - group2_scores.mean()
        direction = f"{x} > {y}" if obs_diff > 0 else f"{y} > {x}"
        new_data = pd.DataFrame([{'Group 1': x, 'Group 2': y, f'{metric}: Group 1': group1_scores.mean(), f'{metric}: Group 2': group2_scores.mean(), 'Observed Difference': obs_diff, 'P-Value': p_value, 'Statistical Significance': 'Significant' if p_value < 0.05 else 'Not Significant', 'Direction': direction}])
        race_sex_df = pd.concat([race_sex_df, new_data], ignore_index=True)
    if model_type == "base":
        race_sex_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/{metric}_scores/race_sex_{metric}.csv')
    elif model_type == "finetuned":
        race_sex_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/{metric}_scores/race_sex_{metric}.csv')

def evaluate_race_age_table_mann_test(test_chexpert_reports, model_type, metric):
    '''
    Method to evaluate the overall accuracy and make intersectional race-age comparisons
    '''
    logger.info(f'Evaluating intersectional race-age comparisons for {metric}')
    race_age_df = pd.DataFrame(columns = ['Group 1', 'Group 2', f'{metric}: Group 1', f'{metric}: Group 2', 'Observed Difference', 'P-Value', 'Statistical Significance', 'Direction'])
    for x, y in combinations(race_age_columns, 2):
        group1_scores = []
        group2_scores = []
        x_parts = x.split('_')
        race_1 = '_'.join(x_parts[:2])
        age_1 = '_'.join(x_parts[2:])
        y_parts = y.split('_')
        race_2 = '_'.join(y_parts[:2])
        age_2 = '_'.join(y_parts[2:])
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
        chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1[age_1] == 1].reset_index(drop = True)
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
        chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2[age_2] == 1].reset_index(drop = True)
        for num in range(10000):
            # Bootstrap sampling for group 1
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            group1_score = calculate_metric(labeled_col_1, original_col_1, metric)
            group1_scores.append(group1_score)

            # Bootstrap sampling for group 2
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            group2_score = calculate_metric(labeled_col_2, original_col_2, metric)
            group2_scores.append(group2_score)

        group1_scores = np.array(group1_scores)
        group2_scores = np.array(group2_scores)
        stat, p_value = mannwhitneyu(group1_scores, group2_scores, alternative='two-sided')
        obs_diff = group1_scores.mean() - group2_scores.mean()
        direction = f"{x} > {y}" if obs_diff > 0 else f"{y} > {x}"
        new_data = pd.DataFrame([{'Group 1': x, 'Group 2': y, f'{metric}: Group 1': group1_scores.mean(), f'{metric}: Group 2': group2_scores.mean(), 'Observed Difference': obs_diff, 'P-Value': p_value, 'Statistical Significance': 'Significant' if p_value < 0.05 else 'Not Significant', 'Direction': direction}])
        race_age_df = pd.concat([race_age_df, new_data], ignore_index=True)
    if model_type == "base":
        race_age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/{metric}_scores/race_age_{metric}.csv')
    elif model_type == "finetuned":
        race_age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/{metric}_scores/race_age_{metric}.csv')

def evaluate_sex_age_table_mann_test(test_chexpert_reports, model_type, metric):
    '''
    Method to evaluate the overall accuracy and make intersectional sex-age comparisons
    '''
    logger.info(f"Evaluating intersectional sex-age comparisons for {metric}")
    sex_age_df = pd.DataFrame(columns = ['Group 1', 'Group 2', f'{metric}: Group 1', f'{metric}: Group 2', 'Observed Difference', 'P-Value', 'Statistical Significance', 'Direction'])
    for x, y in combinations(sex_age_columns, 2):
        group1_scores = []
        group2_scores = []
        x_parts = x.split('_')
        sex_1 = x_parts[0]
        age_1 = '_'.join(x_parts[1:])
        y_parts = y.split('_')
        sex_2 = y_parts[0]
        age_2 = '_'.join(y_parts[1:])
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == sex_1].reset_index(drop = True)
        chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1[age_1] == 1].reset_index(drop = True)
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == sex_2].reset_index(drop = True)
        chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2[age_2] == 1].reset_index(drop = True)
        for num in range(10000):
            # Bootstrap sampling for group 1
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            group1_score = calculate_metric(labeled_col_1, original_col_1, metric)
            group1_scores.append(group1_score)

            # Bootstrap sampling for group 2
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            group2_score = calculate_metric(labeled_col_2, original_col_2, metric)
            group2_scores.append(group2_score)

        group1_scores = np.array(group1_scores)
        group2_scores = np.array(group2_scores)
        stat, p_value = mannwhitneyu(group1_scores, group2_scores, alternative='two-sided')
        obs_diff = group1_scores.mean() - group2_scores.mean()
        direction = f"{x} > {y}" if obs_diff > 0 else f"{y} > {x}"
        new_data = pd.DataFrame([{'Group 1': x, 'Group 2': y, f'{metric}: Group 1': group1_scores.mean(), f'{metric}: Group 2': group2_scores.mean(), 'Observed Difference': obs_diff, 'P-Value': p_value, 'Statistical Significance': 'Significant' if p_value < 0.05 else 'Not Significant', 'Direction': direction}])
        sex_age_df = pd.concat([sex_age_df, new_data], ignore_index=True)
    if model_type == "base":
        sex_age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/{metric}_scores/sex_age_{metric}.csv')
    elif model_type == "finetuned":
        sex_age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/{metric}_scores/sex_age_{metric}.csv')
