import pandas as pd
import numpy as np
from logger import logger
from sklearn.metrics import f1_score
from itertools import combinations
from scipy import stats
from sklearn.metrics import confusion_matrix
from ChexpertPreprocessing import CheXpertPreprocessing
from condition_utils import evaluate_race_condition_table, evaluate_race_age_condition_table, evaluate_race_sex_condition_table, evaluate_age_condition_table

#Set paths and constants
path = '/data/raw_data/chexpert/chexpertplus/df_chexpert_plus_240401.csv'
preprocessed_path = "/home/pr2762@mc.cumc.columbia.edu/CXR_Attribution/splits/preprocessed_chexpert.csv"
original_columns = ['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
       'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']
labeled_columns = [col + '_Labeled' for col in original_columns]
race_columns = ['race_WHITE', 'race_BLACK', 'race_ASIAN', 'race_HISPANIC'] 

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


def llava_evaluation(model_type):
    '''
    Method to generate a clean dataset for evaluation of the llava-rad model
    '''
    if model_type == 'base':
        generated_reports = pd.read_csv("/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/generated_reports.csv")
        labeled_reports = pd.read_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/base_labeled_reports.csv')
    elif model_type == 'finetuned':
        generated_reports = pd.read_csv("/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/generated_reports.csv")
        labeled_reports = pd.read_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/finetuned_labeled_reports.csv')
    labeled_reports['Reports'] = labeled_reports['Reports'].str.replace('"', '', regex=False)
    labeled_reports.columns = [col + '_Labeled' for col in labeled_reports.columns]
    test_data = pd.concat([generated_reports, labeled_reports], axis=1)
    for i, row in test_data.iterrows():
        if row['report'] != row['Reports_Labeled']:
            print(row['report'], row['Reports_Labeled'])
            break
    splits = pd.read_csv(preprocessed_path)
    chexpert_obj = CheXpertPreprocessing(path, splits)
    chexpert_df = chexpert_obj.read_data()
    chexpert_reports = pd.merge(test_data, chexpert_df, on = 'path_to_image', how = 'left')
    return chexpert_reports

def evaluate_f1_score(model_type):
    '''
    Method to evaluate the F1 score for the labeled CheXpert reports
    '''
    test_chexpert_reports = llava_evaluation(model_type)
    nan_proportion = test_chexpert_reports[labeled_columns].isna().mean()
    logger.info(f"Proportion of NaN values in labeled columns: {nan_proportion}")
    #Calculate F1 score for all columns
    test_chexpert_reports[labeled_columns] = test_chexpert_reports[labeled_columns].replace(-1, 0)
    test_chexpert_reports[labeled_columns] = test_chexpert_reports[labeled_columns].fillna(0)
    y_label = test_chexpert_reports[labeled_columns]
    y_true = test_chexpert_reports[original_columns]
    score = f1_score(y_true, y_label, average = 'macro')
    logger.info(f"F1 score for all columns: {score}")
    # Calculate condition table breakdowns
    evaluate_race_condition_table(test_chexpert_reports, model_type)
    evaluate_age_condition_table(test_chexpert_reports, model_type)
    evaluate_sex_condition_table(test_chexpert_reports, model_type)
    evaluate_race_sex_condition_table(test_chexpert_reports, model_type)
    evaluate_race_age_condition_table(test_chexpert_reports, model_type)

evaluate_f1_score("base")
evaluate_f1_score("finetuned")