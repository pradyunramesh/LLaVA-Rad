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

def compute_tpr_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

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

def evaluate_race_condition_table(test_chexpert_reports, model_type):
    # Calculate table for race vs condition
    race_condition_df = pd.DataFrame(columns = ['Condition', 'Race 1', 'Race 2', 'F1 Score: Race 1', 'F1 Score: Race 2', 'P-Value F1', 
    'Statistical Significance F1', 'TPR Score: Race 1', 'TPR Score: Race 2', 'P-Value TPR', 'Statistical Significance TPR',
    'FPR Score: Race 1', 'FPR Score: Race 2', 'P-Value FPR', 'Statistical Significance FPR'])
    for col in range(len(labeled_columns)):
        for x, y in combinations(race_columns, 2):
            #Calculate observed difference in F1 scores, TPR and FPR
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports_1[labeled_columns[col]]
            original_col = chexpert_reports_1[original_columns[col]]
            score_1_og = f1_score(original_col, labeled_col, average = 'binary')
            tpr_1, fpr_1 = compute_tpr_fpr(original_col, labeled_col)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports_2[labeled_columns[col]]
            original_col = chexpert_reports_2[original_columns[col]]
            score_2_og = f1_score(original_col, labeled_col, average = 'binary')
            tpr_2, fpr_2 = compute_tpr_fpr(original_col, labeled_col)

            obs_value = score_1_og - score_2_og
            obs_tpr_diff = tpr_1 - tpr_2
            obs_fpr_diff = fpr_1 - fpr_2

            # Perform bootstrap sampling to evaluate t-test with statistical significance
            metrics = []
            tpr_metrics = []
            fpr_metrics = []
            for num in range(10000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                race1_score = f1_score(original_col_1, labeled_col_1, average = 'binary')
                race1_tpr, race1_fpr = compute_tpr_fpr(original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                race2_score = f1_score(original_col_2, labeled_col_2, average = 'binary')
                race2_tpr, race2_fpr = compute_tpr_fpr(original_col_2, labeled_col_2)

                metrics.append(race1_score - race2_score)
                tpr_metrics.append(race1_tpr - race2_tpr)
                fpr_metrics.append(race1_fpr - race2_fpr)

            metrics = np.array(metrics)
            tpr_metrics = np.array(tpr_metrics)
            fpr_metrics = np.array(fpr_metrics)

            p_value, significance = calculate_p_value(obs_value, metrics, "F1", x, y)
            tpr_p_value, tpr_significance = calculate_p_value(obs_tpr_diff, tpr_metrics, "TPR", x, y)
            fpr_p_value, fpr_significance = calculate_p_value(obs_fpr_diff, fpr_metrics, "FPR", x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Race 1': x, 'Race 2': y, 'F1 Score: Race 1': score_1_og, 
                'F1 Score: Race 2': score_2_og, 'P-Value F1': p_value, 'Statistical Significance F1': significance, 'TPR Score: Race 1': tpr_1, 
                'TPR Score: Race 2': tpr_2, 'P-Value TPR': tpr_p_value, 'Statistical Significance TPR': tpr_significance, 
                'FPR Score: Race 1': fpr_1, 'FPR Score: Race 2': fpr_2, 'P-Value FPR': fpr_p_value, 'Statistical Significance FPR': fpr_significance
            }])
            race_condition_df = pd.concat([race_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        race_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/race_condition_f1_scores.csv')
    elif model_type == "finetuned":
        race_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/race_condition_f1_scores.csv')


def evaluate_sex_condition_table(test_chexpert_reports, model_type):
    ## Calculate table for sex vs condition
    sex_condition_df = pd.DataFrame(columns = ['Condition', 'Sex 1', 'Sex 2', 'F1 Score: Sex 1', 'F1 Score: Sex 2', 'P-Value F1', 
    'Statistical Significance F1', 'TPR Score: Sex 1', 'TPR Score: Sex 2', 'P-Value TPR', 'Statistical Significance TPR',
    'FPR Score: Sex 1', 'FPR Score: Sex 2', 'P-Value FPR', 'Statistical Significance FPR'])
    for col in range(len(labeled_columns)):
        # Calculate observed F1 score
        for x, y in combinations(sex_columns, 2):
            #Calculate observed difference in F1 scores
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == x].reset_index(drop = True)
            labeled_col_1 = chexpert_reports_1[labeled_columns[col]]
            original_col_1 = chexpert_reports_1[original_columns[col]]
            score_1_og = f1_score(original_col_1, labeled_col_1, average = 'binary')
            tpr_1, fpr_1 = compute_tpr_fpr(original_col_1, labeled_col_1)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == y].reset_index(drop = True)
            labeled_col_2 = chexpert_reports_2[labeled_columns[col]]
            original_col_2 = chexpert_reports_2[original_columns[col]]
            score_2_og = f1_score(original_col_2, labeled_col_2, average = 'binary')
            tpr_2, fpr_2 = compute_tpr_fpr(original_col_2, labeled_col_2)

            obs_value = score_1_og - score_2_og
            obs_tpr_diff = tpr_1 - tpr_2
            obs_fpr_diff = fpr_1 - fpr_2

            # Perform bootstrap sampling to evaluate t-test with statistical significance
            metrics = []
            tpr_metrics = []
            fpr_metrics = []
            for num in range(10000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports['sex'] == x].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                score_1 = f1_score(original_col_1, labeled_col_1, average = 'binary')
                sex1_tpr, sex1_fpr = compute_tpr_fpr(original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports['sex'] == y].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                score_2 = f1_score(original_col_2, labeled_col_2, average = 'binary')
                sex2_tpr, sex2_fpr = compute_tpr_fpr(original_col_2, labeled_col_2)

                metrics.append(score_1 - score_2)
                tpr_metrics.append(sex1_tpr - sex2_tpr)
                fpr_metrics.append(sex1_fpr - sex2_fpr)
            
            metrics = np.array(metrics)
            tpr_metrics = np.array(tpr_metrics)
            fpr_metrics = np.array(fpr_metrics)

            p_value, significance = calculate_p_value(obs_value, metrics, "F1", x, y)
            tpr_p_value, tpr_significance = calculate_p_value(obs_tpr_diff, tpr_metrics, "TPR", x, y)
            fpr_p_value, fpr_significance = calculate_p_value(obs_fpr_diff, fpr_metrics, "FPR", x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Sex 1': x, 'Sex 2': y, 'F1 Score: Sex 1': score_1_og, 
                'F1 Score: Sex 2': score_2_og, 'P-Value F1': p_value, 'Statistical Significance F1': significance, 'TPR Score: Sex 1': tpr_1, 
                'TPR Score: Sex 2': tpr_2, 'P-Value TPR': tpr_p_value, 'Statistical Significance TPR': tpr_significance, 
                'FPR Score: Sex 1': fpr_1, 'FPR Score: Sex 2': fpr_2, 'P-Value FPR': fpr_p_value, 'Statistical Significance FPR': fpr_significance
            }])
            sex_condition_df = pd.concat([sex_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        sex_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/sex_condition_f1_scores.csv')
    elif model_type == "finetuned":
        sex_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/sex_condition_f1_scores.csv')

def evaluate_age_condition_table(test_chexpert_reports, model_type):
    ## Calculate table for age vs condition
    age_condition_df = pd.DataFrame(columns = ['Condition', 'Age 1', 'Age 2', 'F1 Score: Age 1', 'F1 Score: Age 2', 'P-Value F1', 
    'Statistical Significance F1', 'TPR Score: Age 1', 'TPR Score: Age 2', 'P-Value TPR', 'Statistical Significance TPR',
    'FPR Score: Age 1', 'FPR Score: Age 2', 'P-Value FPR', 'Statistical Significance FPR'])
    for col in range(len(labeled_columns)):
        # Calculate observed F1 score
        for x, y in combinations(age_columns, 2):
            #Calculate observed difference in F1 scores
            chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports_1[labeled_columns[col]]
            original_col = chexpert_reports_1[original_columns[col]]
            score_1_og = f1_score(original_col, labeled_col, average = 'binary')
            tpr_1, fpr_1 = compute_tpr_fpr(original_col, labeled_col)

            chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports_2[labeled_columns[col]]
            original_col = chexpert_reports_2[original_columns[col]]
            score_2_og = f1_score(original_col, labeled_col, average = 'binary')
            tpr_2, fpr_2 = compute_tpr_fpr(original_col, labeled_col)

            obs_value = score_1_og - score_2_og
            obs_tpr_diff = tpr_1 - tpr_2
            obs_fpr_diff = fpr_1 - fpr_2

            # Perform bootstrap sampling to evaluate t-test with statistical significance
            metrics = []
            tpr_metrics = []
            fpr_metrics = []
            for num in range(10000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[x] == 1].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                age1_score = f1_score(original_col_1, labeled_col_1, average = 'binary')
                age1_tpr, age1_fpr = compute_tpr_fpr(original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[y] == 1].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                age2_score = f1_score(original_col_2, labeled_col_2, average = 'binary')
                age2_tpr, age2_fpr = compute_tpr_fpr(original_col_2, labeled_col_2)

                metrics.append(age1_score - age2_score)
                tpr_metrics.append(age1_tpr - age2_tpr)
                fpr_metrics.append(age1_fpr - age2_fpr)

            metrics = np.array(metrics)
            tpr_metrics = np.array(tpr_metrics)
            fpr_metrics = np.array(fpr_metrics)
            
            p_value, significance = calculate_p_value(obs_value, metrics, "F1", x, y)
            tpr_p_value, tpr_significance = calculate_p_value(obs_tpr_diff, tpr_metrics, "TPR", x, y)
            fpr_p_value, fpr_significance = calculate_p_value(obs_fpr_diff, fpr_metrics, "FPR", x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Age 1': x, 'Age 2': y, 'F1 Score: Age 1': score_1_og, 
                'F1 Score: Age 2': score_2_og, 'P-Value F1': p_value, 'Statistical Significance F1': significance, 'TPR Score: Age 1': tpr_1, 
                'TPR Score: Age 2': tpr_2, 'P-Value TPR': tpr_p_value, 'Statistical Significance TPR': tpr_significance, 
                'FPR Score: Age 1': fpr_1, 'FPR Score: Age 2': fpr_2, 'P-Value FPR': fpr_p_value, 'Statistical Significance FPR': fpr_significance
            }])
            age_condition_df = pd.concat([age_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        age_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/age_condition_f1_scores.csv')
    elif model_type == "finetuned":
        age_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/age_condition_f1_scores.csv')

def evaluate_race_sex_condition_table(test_chexpert_reports, model_type):
    ## Calculate table for race vs sex vs condition
    race_sex_condition_df = pd.DataFrame(columns=[
        'Condition', 'Group 1', 'Group 2',
        'F1 Score: Group 1', 'F1 Score: Group 2', 'P-Value F1', 'Statistical Significance F1',
        'TPR Score: Group 1', 'TPR Score: Group 2', 'P-Value TPR', 'Statistical Significance TPR',
        'FPR Score: Group 1', 'FPR Score: Group 2', 'P-Value FPR', 'Statistical Significance FPR'
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
            score_1_og = f1_score(original_col, labeled_col, average = 'binary')
            tpr_1, fpr_1 = compute_tpr_fpr(original_col, labeled_col)

            y_parts = y.rsplit('_', 1)
            race_2 = y_parts[0]
            sex_2 = y_parts[1]
            chexpert_reports = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
            chexpert_reports = chexpert_reports[chexpert_reports['sex'] == sex_2].reset_index(drop = True)
            labeled_col = chexpert_reports[labeled_columns[col]]
            original_col = chexpert_reports[original_columns[col]]
            score_2_og = f1_score(original_col, labeled_col, average = 'binary')
            tpr_2, fpr_2 = compute_tpr_fpr(original_col, labeled_col)

            obs_value = score_1_og - score_2_og
            obs_tpr_diff = tpr_1 - tpr_2
            obs_fpr_diff = fpr_1 - fpr_2

            metrics = []
            tpr_metrics = []
            fpr_metrics = []
            for num in range(1000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
                chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1['sex'] == sex_1].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                group1_score = f1_score(original_col_1, labeled_col_1, average = 'binary')
                group1_tpr, group1_fpr = compute_tpr_fpr(original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
                chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2['sex'] == sex_2].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                group2_score = f1_score(original_col_2, labeled_col_2, average = 'binary')
                group2_tpr, group2_fpr = compute_tpr_fpr(original_col_2, labeled_col_2)

                metrics.append(group1_score - group2_score)
                tpr_metrics.append(group1_tpr - group2_tpr)
                fpr_metrics.append(group1_fpr - group2_fpr)
            
            metrics = np.array(metrics)
            tpr_metrics = np.array(tpr_metrics)
            fpr_metrics = np.array(fpr_metrics)
            
            p_value, significance = calculate_p_value(obs_value, metrics, "F1", x, y)
            tpr_p_value, tpr_significance = calculate_p_value(obs_tpr_diff, tpr_metrics, "TPR", x, y)
            fpr_p_value, fpr_significance = calculate_p_value(obs_fpr_diff, fpr_metrics, "FPR", x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Group 1': x, 'Group 2': y, 'F1 Score: Group 1': score_1_og, 
                'F1 Score: Group 2': score_2_og, 'P-Value F1': p_value, 'Statistical Significance F1': significance, 'TPR Score: Group 1': tpr_1, 
                'TPR Score: Group 2': tpr_2, 'P-Value TPR': tpr_p_value, 'Statistical Significance TPR': tpr_significance, 
                'FPR Score: Group 1': fpr_1, 'FPR Score: Group 2': fpr_2, 'P-Value FPR': fpr_p_value, 'Statistical Significance FPR': fpr_significance
            }])
            race_sex_condition_df = pd.concat([race_sex_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        race_sex_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/race_sex_condition_f1_scores.csv')
    elif model_type == "finetuned":
        race_sex_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/race_sex_condition_f1_scores.csv')

def evaluate_race_age_condition_table(test_chexpert_reports, model_type):
    ## Calculate table for race vs age vs condition
    race_age_condition_df = pd.DataFrame(columns=[
        'Condition', 'Group 1', 'Group 2',
        'F1 Score: Group 1', 'F1 Score: Group 2', 'P-Value F1', 'Statistical Significance F1',
        'TPR Score: Group 1', 'TPR Score: Group 2', 'P-Value TPR', 'Statistical Significance TPR',
        'FPR Score: Group 1', 'FPR Score: Group 2', 'P-Value FPR', 'Statistical Significance FPR'
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
            score_1_og = f1_score(original_col, labeled_col, average = 'binary')
            tpr_1, fpr_1 = compute_tpr_fpr(original_col, labeled_col)

            y_parts = y.split('_')
            race_2 = '_'.join(y_parts[:2])
            age_2 = '_'.join(y_parts[2:])
            chexpert_reports = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
            chexpert_reports = chexpert_reports[chexpert_reports[age_2] == 1].reset_index(drop = True)
            labeled_col = chexpert_reports[labeled_columns[col]]
            original_col = chexpert_reports[original_columns[col]]
            score_2_og = f1_score(original_col, labeled_col, average = 'binary')
            tpr_2, fpr_2 = compute_tpr_fpr(original_col, labeled_col)

            obs_value = score_1_og - score_2_og
            obs_tpr_diff = tpr_1 - tpr_2
            obs_fpr_diff = fpr_1 - fpr_2

            metrics = []
            tpr_metrics = []
            fpr_metrics = []
            for num in range(1000):
                chexpert_reports_1 = test_chexpert_reports[test_chexpert_reports[race_1] == 1].reset_index(drop = True)
                chexpert_reports_1 = chexpert_reports_1[chexpert_reports_1[age_1] == 1].reset_index(drop = True)
                chexpert_reports_1_sample = chexpert_reports_1.sample(frac = 1, replace = True, random_state=num)
                labeled_col_1 = chexpert_reports_1_sample[labeled_columns[col]]
                original_col_1 = chexpert_reports_1_sample[original_columns[col]]
                group1_score = f1_score(original_col_1, labeled_col_1, average = 'binary')
                group1_tpr, group1_fpr = compute_tpr_fpr(original_col_1, labeled_col_1)

                chexpert_reports_2 = test_chexpert_reports[test_chexpert_reports[race_2] == 1].reset_index(drop = True)
                chexpert_reports_2 = chexpert_reports_2[chexpert_reports_2[age_2] == 1].reset_index(drop = True)
                chexpert_reports_2_sample = chexpert_reports_2.sample(frac = 1, replace = True, random_state=num)
                labeled_col_2 = chexpert_reports_2_sample[labeled_columns[col]]
                original_col_2 = chexpert_reports_2_sample[original_columns[col]]
                group2_score = f1_score(original_col_2, labeled_col_2, average = 'binary')
                group2_tpr, group2_fpr = compute_tpr_fpr(original_col_2, labeled_col_2)

                metrics.append(group1_score - group2_score)
                tpr_metrics.append(group1_tpr - group2_tpr)
                fpr_metrics.append(group1_fpr - group2_fpr)
            
            metrics = np.array(metrics)
            tpr_metrics = np.array(tpr_metrics)
            fpr_metrics = np.array(fpr_metrics)
            
            p_value, significance = calculate_p_value(obs_value, metrics, "F1", x, y)
            tpr_p_value, tpr_significance = calculate_p_value(obs_tpr_diff, tpr_metrics, "TPR", x, y)
            fpr_p_value, fpr_significance = calculate_p_value(obs_fpr_diff, fpr_metrics, "FPR", x, y)

            new_data = pd.DataFrame([{'Condition': original_columns[col], 'Group 1': x, 'Group 2': y, 'F1 Score: Group 1': score_1_og, 
                'F1 Score: Group 2': score_2_og, 'P-Value F1': p_value, 'Statistical Significance F1': significance, 'TPR Score: Group 1': tpr_1, 
                'TPR Score: Group 2': tpr_2, 'P-Value TPR': tpr_p_value, 'Statistical Significance TPR': tpr_significance, 
                'FPR Score: Group 1': fpr_1, 'FPR Score: Group 2': fpr_2, 'P-Value FPR': fpr_p_value, 'Statistical Significance FPR': fpr_significance
            }])
            race_age_condition_df = pd.concat([race_age_condition_df, new_data], ignore_index=True)
    if model_type == "base":
        race_age_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/eval/f1_scores/race_age_condition_f1_scores.csv')
    elif model_type == "finetuned":
        race_age_condition_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/f1_scores/race_age_condition_f1_scores.csv')