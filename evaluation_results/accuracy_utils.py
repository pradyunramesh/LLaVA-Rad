import pandas as pd
import numpy as np
from logger import logger
from itertools import combinations

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

def calculate_accuracy(y_pred, y_true):
    '''
    Method to calculate accuracy given predicted and true labels
    '''
    y_pred = y_pred.to_numpy()
    y_true = y_true.to_numpy()
    label_accuracies = (y_pred == y_true).mean(axis=0)
    average_accuracy = label_accuracies.mean()
    return average_accuracy

def evaluate_race_accuracy_table(test_chexpert_reports, model_type):
    '''
    Method to evaluate the overall accuracy and make race-based comparisons
    '''
    race_df = pd.DataFrame(columns = ['Race 1', 'Race 2', 'Accuracy: Race 1', 'Accuracy: Race 2', 'P-Value', 'Statistical Significance'])
    for x, y in combinations(race_columns, 2):
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
        labeled_col = chexpert_reports_1[labeled_columns]
        original_col = chexpert_reports_1[original_columns]
        score_1_og = calculate_accuracy(labeled_col, original_col)

        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
        labeled_col = chexpert_reports_2[labeled_columns]
        original_col = chexpert_reports_2[original_columns]
        score_2_og = calculate_accuracy(labeled_col, original_col)

        obs_value = score_1_og - score_2_og
        scores = []
        for num in range(10000):
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            race1_score = calculate_accuracy(labeled_col_1, original_col_1)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            race2_score = calculate_accuracy(labeled_col_2, original_col_2)

            scores.append(race1_score - race2_score)

        scores = np.array(scores)
        p_value, significance = calculate_p_value(obs_value, scores, "accuracy", x, y)

        new_data = pd.DataFrame([{'Race 1': x, 'Race 2': y, 'Accuracy: Race 1': score_1_og, 'Accuracy: Race 2': score_2_og, 'P-Value': p_value, 'Statistical Significance': significance}])
        race_df = pd.concat([race_df, new_data], ignore_index=True)
    if model_type == "base":
        race_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/accuracy_scores/race_accuracy.csv')
    elif model_type == "finetuned":
        race_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/accuracy_scores/race_accuracy.csv')

def evaluate_sex_accuracy_table(test_chexpert_reports, model_type):
    '''
    Method to evaluate the overall accuracy and make sex-based comparisons
    '''
    sex_df = pd.DataFrame(columns = ['Sex 1', 'Sex 2', 'Accuracy: Sex 1', 'Accuracy: Sex 2', 'P-Value', 'Statistical Significance'])
    for x, y in combinations(sex_columns, 2):
        # Using 'sex' column for filtering as shown in condition_utils.py
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == x].reset_index(drop = True)
        labeled_col = chexpert_reports_1[labeled_columns]
        original_col = chexpert_reports_1[original_columns]
        score_1_og = calculate_accuracy(labeled_col, original_col)

        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == y].reset_index(drop = True)
        labeled_col = chexpert_reports_2[labeled_columns]
        original_col = chexpert_reports_2[original_columns]
        score_2_og = calculate_accuracy(labeled_col, original_col)

        obs_value = score_1_og - score_2_og
        scores = []
        for num in range(10000):
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == x].reset_index(drop = True)
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            sex1_score = calculate_accuracy(labeled_col_1, original_col_1)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == y].reset_index(drop = True)
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            sex2_score = calculate_accuracy(labeled_col_2, original_col_2)

            scores.append(sex1_score - sex2_score)

        scores = np.array(scores)
        p_value, significance = calculate_p_value(obs_value, scores, "accuracy", x, y)

        new_data = pd.DataFrame([{'Sex 1': x, 'Sex 2': y, 'Accuracy: Sex 1': score_1_og, 'Accuracy: Sex 2': score_2_og, 'P-Value': p_value, 'Statistical Significance': significance}])
        sex_df = pd.concat([sex_df, new_data], ignore_index=True)
    if model_type == "base":
        sex_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/accuracy_scores/sex_accuracy.csv')
    elif model_type == "finetuned":
        sex_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/accuracy_scores/sex_accuracy.csv')

def evaluate_age_accuracy_table(test_chexpert_reports, model_type):
    '''
    Method to evaluate the overall accuracy and make age-based comparisons
    '''
    age_df = pd.DataFrame(columns = ['Age 1', 'Age 2', 'Accuracy: Age 1', 'Accuracy: Age 2', 'P-Value', 'Statistical Significance'])
    for x, y in combinations(age_columns, 2):
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
        labeled_col = chexpert_reports_1[labeled_columns]
        original_col = chexpert_reports_1[original_columns]
        score_1_og = calculate_accuracy(labeled_col, original_col)

        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
        labeled_col = chexpert_reports_2[labeled_columns]
        original_col = chexpert_reports_2[original_columns]
        score_2_og = calculate_accuracy(labeled_col, original_col)

        obs_value = score_1_og - score_2_og
        scores = []
        for num in range(10000):
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            age1_score = calculate_accuracy(labeled_col_1, original_col_1)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            age2_score = calculate_accuracy(labeled_col_2, original_col_2)

            scores.append(age1_score - age2_score)

        scores = np.array(scores)
        p_value, significance = calculate_p_value(obs_value, scores, "accuracy", x, y)

        new_data = pd.DataFrame([{'Age 1': x, 'Age 2': y, 'Accuracy: Age 1': score_1_og, 'Accuracy: Age 2': score_2_og, 'P-Value': p_value, 'Statistical Significance': significance}])
        age_df = pd.concat([age_df, new_data], ignore_index=True)
    if model_type == "base":
        age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/accuracy_scores/age_accuracy.csv')
    elif model_type == "finetuned":
        age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/accuracy_scores/age_accuracy.csv')

def evaluate_race_sex_accuracy_table(test_chexpert_reports, model_type):
    '''
    Method to evaluate the overall accuracy and make intersectional race-sex comparisons
    '''
    race_sex_df = pd.DataFrame(columns = ['Group 1', 'Group 2', 'Accuracy: Group 1', 'Accuracy: Group 2', 'P-Value', 'Statistical Significance'])
    for x, y in combinations(race_sex_columns, 2):
        # Extract race and sex from combined column names
        x_parts = x.rsplit('_', 1)
        race_1 = x_parts[0]
        sex_1 = x_parts[1]
        
        # Filter reports by both race and sex
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
        chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1['sex'] == sex_1].reset_index(drop = True)
        labeled_col = chexpert_reports_1[labeled_columns]
        original_col = chexpert_reports_1[original_columns]
        score_1_og = calculate_accuracy(labeled_col, original_col)

        # Same for second group
        y_parts = y.rsplit('_', 1)
        race_2 = y_parts[0]
        sex_2 = y_parts[1]
        
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
        chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2['sex'] == sex_2].reset_index(drop = True)
        labeled_col = chexpert_reports_2[labeled_columns]
        original_col = chexpert_reports_2[original_columns]
        score_2_og = calculate_accuracy(labeled_col, original_col)

        obs_value = score_1_og - score_2_og
        scores = []
        for num in range(10000):
            # Bootstrap sampling for group 1
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
            chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1['sex'] == sex_1].reset_index(drop = True)
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            group1_score = calculate_accuracy(labeled_col_1, original_col_1)

            # Bootstrap sampling for group 2
            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
            chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2['sex'] == sex_2].reset_index(drop = True)
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            group2_score = calculate_accuracy(labeled_col_2, original_col_2)

            scores.append(group1_score - group2_score)

        scores = np.array(scores)
        p_value, significance = calculate_p_value(obs_value, scores, "accuracy", x, y)

        new_data = pd.DataFrame([{'Group 1': x, 'Group 2': y, 'Accuracy: Group 1': score_1_og, 'Accuracy: Group 2': score_2_og, 'P-Value': p_value, 'Statistical Significance': significance}])
        race_sex_df = pd.concat([race_sex_df, new_data], ignore_index=True)
    if model_type == "base":
        race_sex_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/accuracy_scores/race_sex_accuracy.csv')
    elif model_type == "finetuned":
        race_sex_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/accuracy_scores/race_sex_accuracy.csv')
        
def evaluate_race_age_accuracy_table(test_chexpert_reports, model_type):
    '''
    Method to evaluate the overall accuracy and make intersectional race-age comparisons
    '''
    race_age_df = pd.DataFrame(columns = ['Group 1', 'Group 2', 'Accuracy: Group 1', 'Accuracy: Group 2', 'P-Value', 'Statistical Significance'])
    for x, y in combinations(race_age_columns, 2):
        # Extract race and age from combined column names - different parsing pattern than race_sex
        x_parts = x.split('_')
        race_1 = '_'.join(x_parts[:2])  # e.g., race_WHITE
        age_1 = '_'.join(x_parts[2:])   # e.g., anchor_age_0.0
        
        # Filter reports by both race and age
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
        chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1[age_1] == 1].reset_index(drop = True)
        labeled_col = chexpert_reports_1[labeled_columns]
        original_col = chexpert_reports_1[original_columns]
        score_1_og = calculate_accuracy(labeled_col, original_col)

        # Same for second group
        y_parts = y.split('_')
        race_2 = '_'.join(y_parts[:2])
        age_2 = '_'.join(y_parts[2:])
        
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
        chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2[age_2] == 1].reset_index(drop = True)
        labeled_col = chexpert_reports_2[labeled_columns]
        original_col = chexpert_reports_2[original_columns]
        score_2_og = calculate_accuracy(labeled_col, original_col)

        obs_value = score_1_og - score_2_og
        scores = []
        for num in range(10000):
            # Bootstrap sampling for group 1
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
            chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1[age_1] == 1].reset_index(drop = True)
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            group1_score = calculate_accuracy(labeled_col_1, original_col_1)

            # Bootstrap sampling for group 2
            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
            chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2[age_2] == 1].reset_index(drop = True)
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            group2_score = calculate_accuracy(labeled_col_2, original_col_2)

            scores.append(group1_score - group2_score)

        scores = np.array(scores)
        p_value, significance = calculate_p_value(obs_value, scores, "accuracy", x, y)

        new_data = pd.DataFrame([{'Group 1': x, 'Group 2': y, 'Accuracy: Group 1': score_1_og, 'Accuracy: Group 2': score_2_og, 'P-Value': p_value, 'Statistical Significance': significance}])
        race_age_df = pd.concat([race_age_df, new_data], ignore_index=True)
    if model_type == "base":
        race_age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/accuracy_scores/race_age_accuracy.csv')
    elif model_type == "finetuned":
        race_age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/accuracy_scores/race_age_accuracy.csv')

def evaluate_sex_age_accuracy_table(test_chexpert_reports, model_type):
    '''
    Method to evaluate the overall accuracy and make intersectional sex-age comparisons
    '''
    sex_age_df = pd.DataFrame(columns = ['Group 1', 'Group 2', 'Accuracy: Group 1', 'Accuracy: Group 2', 'P-Value', 'Statistical Significance'])
    for x, y in combinations(sex_age_columns, 2):
        # Extract sex and age from combined column names
        x_parts = x.split('_')
        sex_1 = x_parts[0]              # e.g., Female
        age_1 = '_'.join(x_parts[1:])   # e.g., anchor_age_0.0
        
        # Filter reports by both sex and age
        chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == sex_1].reset_index(drop = True)
        chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1[age_1] == 1].reset_index(drop = True)
        labeled_col = chexpert_reports_1[labeled_columns]
        original_col = chexpert_reports_1[original_columns]
        score_1_og = calculate_accuracy(labeled_col, original_col)

        # Same for second group
        y_parts = y.split('_')
        sex_2 = y_parts[0]
        age_2 = '_'.join(y_parts[1:])
        
        chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == sex_2].reset_index(drop = True)
        chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2[age_2] == 1].reset_index(drop = True)
        labeled_col = chexpert_reports_2[labeled_columns]
        original_col = chexpert_reports_2[original_columns]
        score_2_og = calculate_accuracy(labeled_col, original_col)

        obs_value = score_1_og - score_2_og
        scores = []
        for num in range(10000):
            # Bootstrap sampling for group 1
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == sex_1].reset_index(drop = True)
            chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1[age_1] == 1].reset_index(drop = True)
            chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
            labeled_col_1 = chexpert_reports_1_sample[labeled_columns]
            original_col_1 = chexpert_reports_1_sample[original_columns]
            group1_score = calculate_accuracy(labeled_col_1, original_col_1)

            # Bootstrap sampling for group 2
            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == sex_2].reset_index(drop = True)
            chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2[age_2] == 1].reset_index(drop = True)
            chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
            labeled_col_2 = chexpert_reports_2_sample[labeled_columns]
            original_col_2 = chexpert_reports_2_sample[original_columns]
            group2_score = calculate_accuracy(labeled_col_2, original_col_2)

            scores.append(group1_score - group2_score)

        scores = np.array(scores)
        p_value, significance = calculate_p_value(obs_value, scores, "accuracy", x, y)

        new_data = pd.DataFrame([{'Group 1': x, 'Group 2': y, 'Accuracy: Group 1': score_1_og, 'Accuracy: Group 2': score_2_og, 'P-Value': p_value, 'Statistical Significance': significance}])
        sex_age_df = pd.concat([sex_age_df, new_data], ignore_index=True)
    if model_type == "base":
        sex_age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/accuracy_scores/sex_age_accuracy.csv')
    elif model_type == "finetuned":
        sex_age_df.to_csv(f'/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/accuracy_scores/sex_age_accuracy.csv')
