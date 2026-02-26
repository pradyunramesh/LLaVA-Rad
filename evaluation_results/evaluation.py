import pandas as pd
import numpy as np
from logger import logger
from sklearn.metrics import f1_score
from itertools import combinations
from scipy import stats
from sklearn.metrics import confusion_matrix
from ChexpertPreprocessing import CheXpertPreprocessing
from condition_utils import evaluate_race_condition_table, evaluate_race_age_condition_table, evaluate_race_sex_condition_table, evaluate_age_condition_table, evaluate_sex_condition_table
from no_condition_utils import evaluate_race_table, evaluate_sex_table, evaluate_age_table, evaluate_race_sex_table, evaluate_race_age_table, evaluate_sex_age_table
from mann_whitney_utils import evaluate_race_table_mann_test, evaluate_sex_table_mann_test, evaluate_age_table_mann_test, evaluate_race_sex_table_mann_test, evaluate_race_age_table_mann_test, evaluate_sex_age_table_mann_test
from quantile_analysis import quantile_analysis_accuracy, quantile_analysis_demo, condition_attributions_groups

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
        generated_reports = pd.read_csv("/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/generated_reports.csv")
        labeled_reports = pd.read_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_base/base_labeled_reports.csv')
    elif model_type == 'finetuned':
        generated_reports = pd.read_csv("/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/generated_reports.csv")
        labeled_reports = pd.read_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/finetuned_labeled_reports.csv')
    elif model_type == 'finetuned_gpt':
        generated_reports = pd.read_csv("/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned_final/generated_reports.csv")
        labeled_reports = pd.read_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned_final/labeled_reports.csv')
    labeled_reports['Reports'] = labeled_reports['Reports'].str.replace('"', '', regex=False)
    labeled_reports.columns = [col + '_Labeled' for col in labeled_reports.columns]
    test_data = pd.concat([generated_reports, labeled_reports], axis=1)
    for i, row in test_data.iterrows():
        if row['report'] != row['Reports_Labeled']:
            print(row['report'], row['Reports_Labeled'])
    del test_data['report']
    splits = pd.read_csv(preprocessed_path)
    chexpert_obj = CheXpertPreprocessing(path, splits)
    chexpert_df = chexpert_obj.read_data()
    chexpert_reports = pd.merge(test_data, chexpert_df, on = 'path_to_image', how = 'left')
    columns = chexpert_reports.columns.to_list()
    #pd.Series(columns).to_csv('csv_files/llava_columns.csv', index=False)
    #chexpert_reports.to_csv('/home/pr2762@mc.cumc.columbia.edu/chexpert_test_data.csv', index=False)
    return chexpert_reports

def calculate_accuracy(y_pred, y_true):
    '''
    Method to calculate accuracy given predicted and true labels
    '''
    y_pred = y_pred.to_numpy()
    y_true = y_true.to_numpy()
    label_accuracies = (y_pred == y_true).mean(axis=0)
    average_accuracy = label_accuracies.mean()
    return average_accuracy

def evaluate_f1_score(model_type, metrics):
    '''
    Method to evaluate the F1 score for the labeled CheXpert reports
    '''
    test_chexpert_reports = llava_evaluation(model_type)
    nan_proportion = test_chexpert_reports[labeled_columns].isna().mean()
    test_chexpert_reports[labeled_columns] = test_chexpert_reports[labeled_columns].replace(-1, 0)
    test_chexpert_reports[labeled_columns] = test_chexpert_reports[labeled_columns].fillna(0)

    y_label = test_chexpert_reports[labeled_columns]
    y_true = test_chexpert_reports[original_columns]
    macro_score = f1_score(y_true, y_label, average = 'macro')
    micro_score = f1_score(y_true, y_label, average = 'micro')
    accuracy = calculate_accuracy(y_label, y_true)

    #logger.info(f"Proportion of NaN values in labeled columns: {nan_proportion}")
    logger.info(f"Macro F1 score for all columns: {macro_score}")
    logger.info(f"Micro F1 score for all columns: {micro_score}")
    logger.info(f"Accuracy for all columns: {accuracy}")

    disease_based = False
    for metric in metrics:
        if metric == "accuracy" or metric == "micro-F1" or metric == "macro-F1":
            evaluate_race_table(test_chexpert_reports, model_type, metric)
            evaluate_sex_table(test_chexpert_reports, model_type, metric)
            evaluate_age_table(test_chexpert_reports, model_type, metric)
            evaluate_race_sex_table(test_chexpert_reports, model_type, metric)
            evaluate_race_age_table(test_chexpert_reports, model_type, metric)
            evaluate_sex_age_accuracy_table(test_chexpert_reports, model_type)
        elif metric == "mann-whitney-accuracy" or metric == "mann-whitney-micro-F1" or metric == "mann-whitney-macro-F1":
            evaluate_race_table_mann_test(test_chexpert_reports, model_type, metric)
            evaluate_sex_table_mann_test(test_chexpert_reports, model_type, metric)
            evaluate_age_table_mann_test(test_chexpert_reports, model_type, metric)
            evaluate_race_sex_table_mann_test(test_chexpert_reports, model_type, metric)
            evaluate_race_age_table_mann_test(test_chexpert_reports, model_type, metric)
            evaluate_sex_age_accuracy_table_mann_test(test_chexpert_reports, model_type)
        elif metric == "quantile-analysis":
            if disease_based:
                results = []
                for i, col in enumerate(original_columns):
                    print(original_columns[i] + " " + labeled_columns[i])
                    condition_results = quantile_analysis_accuracy(test_chexpert_reports, [original_columns[i]], [labeled_columns[i]], disease_based = True)
                    if isinstance(condition_results, dict):
                        results_flat = []
                        for var, cramers_value in condition_results.items():
                            results_flat.append({
                                'condition': original_columns[i],
                                'demographic_variable': var,
                                'cramers_v': cramers_values
                            })
                        results_df = pd.DataFrame(results_flat)
                        results.append(results_df)
                combined_results = pd.concat(results, ignore_index=True)
                combined_results.to_csv(f'csv_files/cramersV_results_diseaseBased.csv', index=False)
            else:
                quantile_analysis_accuracy(test_chexpert_reports, original_columns, labeled_columns, disease_based = False)
        elif metric == "quantile-demographics":
            quantile_analysis_demo(test_chexpert_reports, bootstrapping = True)
        elif metric == "condition-attributions":
            condition_attributions_groups(test_chexpert_reports)
        else:
            evaluate_sex_condition_table(test_chexpert_reports, model_type, metric)
            evaluate_race_condition_table(test_chexpert_reports, model_type, metric)
            evaluate_age_condition_table(test_chexpert_reports, model_type, metric)
            evaluate_race_sex_condition_table(test_chexpert_reports, model_type, metric)
            evaluate_race_age_condition_table(test_chexpert_reports, model_type, metric)

metrics = ["condition-attributions"]
evaluate_f1_score("finetuned_gpt", metrics)
#llava_evaluation("finetuned_gpt")