import pandas as pd
import numpy as np
from logger import logger
from sklearn.metrics import f1_score
from itertools import combinations
from scipy import stats
from sklearn.metrics import confusion_matrix
from ChexpertPreprocessing import CheXpertPreprocessing
from condition_utils import evaluate_race_condition_table, evaluate_race_age_condition_table, evaluate_race_sex_condition_table, evaluate_age_condition_table, evaluate_sex_condition_table

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

def llava_evaluation(model_type):
    '''
    Method to generate a clean dataset for evaluation of the llava-rad model
    '''
    if model_type == 'base':
        generated_reports = pd.read_csv("/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base_final/generated_reports.csv")
        labeled_reports = pd.read_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base_final/base_labeled_reports.csv')
    elif model_type == 'finetuned':
        generated_reports = pd.read_csv("/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned_final/generated_reports.csv")
        labeled_reports = pd.read_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned_final/finetuned_labeled_reports.csv')
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
    print(chexpert_reports['loss'].isna().sum())
    return chexpert_reports

def evaluate_f1_score(model_type):
    '''
    Method to evaluate the F1 score for the labeled CheXpert reports
    '''
    test_chexpert_reports = llava_evaluation(model_type)
    nan_proportion = test_chexpert_reports[labeled_columns].isna().mean()
    logger.info(f"Proportion of NaN values in labeled columns: {nan_proportion}")

    test_chexpert_reports[labeled_columns] = test_chexpert_reports[labeled_columns].replace(-1, 0)
    test_chexpert_reports[labeled_columns] = test_chexpert_reports[labeled_columns].fillna(0)
    y_label = test_chexpert_reports[labeled_columns]
    y_true = test_chexpert_reports[original_columns]
    score = f1_score(y_true, y_label, average = 'macro')
    logger.info(f"F1 score for all columns: {score}")

    metrics = ["loss"]
    for metric in metrics:
        if model_type == "base":
            evaluate_sex_condition_table(test_chexpert_reports, model_type, metric)
        else:
            evaluate_race_condition_table(test_chexpert_reports, model_type, metric)
        # evaluate_age_condition_table(test_chexpert_reports, model_type, metric)
            evaluate_sex_condition_table(test_chexpert_reports, model_type, metric)
        # evaluate_race_sex_condition_table(test_chexpert_reports, model_type)
        # evaluate_race_age_condition_table(test_chexpert_reports, model_type)

evaluate_f1_score("base")
evaluate_f1_score("finetuned")
