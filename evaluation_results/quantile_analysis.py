import itertools
import pandas as pd
import numpy as np
from logger import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

race_columns = ['race_WHITE', 'race_BLACK', 'race_ASIAN', 'race_HISPANIC']
age_columns = ['anchor_age_0.0','anchor_age_1.0','anchor_age_2.0','anchor_age_3.0','anchor_age_4.0']
original_columns = ['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
       'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']
labeled_columns = [col + '_Labeled' for col in original_columns]

def calculate_accuracy(y_pred, y_true):
    '''
    Method to calculate accuracy given predicted and true labels
    '''
    y_pred = y_pred.to_numpy()
    y_true = y_true.to_numpy()
    label_accuracies = (y_pred == y_true).mean(axis=0)
    average_accuracy = label_accuracies.mean()
    return average_accuracy

def calculate_p_value(obs_value, metrics, score_type, x, y):
    if obs_value >= 0:
        p_one_sided = np.sum(metrics <= 0) / len(metrics)
    else:
        p_one_sided = np.sum(metrics >= 0) / len(metrics)
    significance = f"No statistical significance for {score_type} score"
    if p_one_sided <0.05:
        if obs_value >= 0:
            significance = f"{y} has a significantly higher {score_type} score than {x}"
        else:
            significance = f"{x} has a significantly higher {score_type} score than {y}"
    return p_one_sided, significance

def quantile_analysis(test_df):
    test_df["race"] = test_df[race_columns].idxmax(axis=1)
    test_df["age"] = test_df[age_columns].idxmax(axis=1)
    test_df["demo_group"] = test_df["race"] + "_" + test_df["age"] + "_" + test_df["sex"]
    metrics = (
        test_df.groupby("demo_group")
        .apply(
            lambda x: pd.Series(
                {
                    "n": len(x),
                    "accuracy": calculate_accuracy(x[labeled_columns], x[original_columns])
                }
            )
        )
    )
    metrics["acc_quantile"] = pd.qcut(metrics["accuracy"], q=5, labels=False)
    lowest_quantile = metrics[metrics["acc_quantile"] == 0].index.tolist()
    highest_quantile = metrics[metrics["acc_quantile"] == metrics["acc_quantile"].max()].index.tolist()
    print(lowest_quantile)
    print(highest_quantile)
    bootstrapping_quantile_analysis(test_df, lowest_quantile, highest_quantile)

def bootstrapping_quantile_analysis(test_df, lowest_quantile, highest_quantile):
    df = pd.DataFrame(columns = ['Group 1', 'Group 2', 'Accuracy: Group 1', 'Accuracy: Group 2', 
    'P-One-Sided', 'Significance-One-Sided'])
    scores = []
    scores_x = []
    scores_y = []
    test_x = test_df[test_df["demo_group"].isin(lowest_quantile)].reset_index(drop=True)
    test_y = test_df[test_df["demo_group"].isin(highest_quantile)].reset_index(drop=True)
    x_observed = calculate_accuracy(test_x[labeled_columns], test_x[original_columns])
    y_observed = calculate_accuracy(test_y[labeled_columns], test_y[original_columns])
    print(x_observed)
    print(y_observed)
    obs_score = y_observed - x_observed

    for num in range(10000):
        test_x_sample = test_x.sample(frac=1, replace=True, random_state=num)
        x_score = calculate_accuracy(test_x_sample[labeled_columns], test_x_sample[original_columns])
        scores_x.append(x_score)

        test_y_sample = test_y.sample(frac=1, replace=True, random_state=num)
        y_score = calculate_accuracy(test_y_sample[labeled_columns], test_y_sample[original_columns])
        scores.append(y_score - x_score)
        scores_y.append(y_score)
            
    scores = np.array(scores)
    lower, upper = np.percentile(scores, [2.5, 97.5])
    lower2, upper2 = np.percentile(scores_x, [2.5, 97.5])
    lower3, upper3 = np.percentile(scores_y, [2.5, 97.5])
    print(lower)
    print(upper)
    print(lower2)
    print(upper2)
    print(lower3)
    print(upper3)
    p_one_sided, significance_one_sided = calculate_p_value(obs_score, scores, "accuracy", 'Group 1', 'Group 2')
    data = pd.DataFrame([{'Group 1': lowest_quantile, 'Group 2': highest_quantile, 'Accuracy: Group 1': x_observed, 'Accuracy: Group 2': y_observed,
    'P-One-Sided': p_one_sided, 'Significance-One-Sided': significance_one_sided}])
    df = pd.concat([data, df], ignore_index=True)
    
    df.to_csv('/home/pr2762@mc.cumc.columbia.edu/LLaVA-Rad/evaluation_results/llavarad_finetuned/eval/quantile_accuracy_scores_race.csv')

def agg_stats(subset_df):
    total_correct = (subset_df["accuracy"] * subset_df["n"]).sum()
    total_n = subset_df["n"].sum()
    return total_correct, total_n

#Do a difference between all the groups in the lowest/highest quantile
#Also do a population based difference instead of a demographic based difference to see if we can form statistically significant groups over there and then do a
# comparison between the lowest and the highest quantile groups.
