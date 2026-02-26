import shap
import itertools
import pandas as pd
import numpy as np
import scipy.stats as ss
from logger import logger
from sklearn.metrics import accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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

def quantile_analysis_demo(test_df, bootstrapping = False):
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
    metrics["acc_quantile"] = pd.qcut(metrics["accuracy"], q=10, labels=False, duplicates = 'drop')
    quantiles = metrics["acc_quantile"].unique()
    lowest_quantile = metrics[metrics["acc_quantile"] == 0].index.tolist()
    highest_quantile = metrics[metrics["acc_quantile"] == metrics["acc_quantile"].max()].index.tolist()
    test_df = test_df.merge(metrics[["acc_quantile"]], left_on="demo_group", right_index = True, how = "left")
    bootstrapping_dfs = []
    if bootstrapping:
        for q1, q2 in itertools.combinations(quantiles, 2):
            q1_groups = metrics[metrics["acc_quantile"] == q1].index.tolist()
            q2_groups = metrics[metrics["acc_quantile"] == q2].index.tolist()
            bootstrapping_dfs.append(bootstrapping_quantile_analysis(test_df, q1_groups, q2_groups, q1, q2))
        final_df = pd.concat(bootstrapping_dfs, ignore_index=True)
        final_df.to_csv('csv_files/Bootstrapping_Quantile_Demographics.csv', index=False)

def condition_attributions_groups(test_df):
    for i, col in enumerate(original_columns):
        test_df[f"{col} Accuracy"] = (test_df[labeled_columns[i]] == test_df[original_columns[i]]).astype(int)
        condition_accuracy = test_df[f"{col} Accuracy"].mean()
        print(f"Accuracy for condition: {col} is {condition_accuracy}")
    test_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/condition_attributions_data.csv', index=False)
    return test_df

def quantile_analysis_accuracy(test_df, true_col, pred_col, disease_based = False, shapley = False):
    #Extract the necessary columns and create demographic groups
    test_df["race"] = test_df[race_columns].idxmax(axis=1)
    test_df["age"] = test_df[age_columns].idxmax(axis=1)
    test_df["demo_group"] = test_df["race"] + "_" + test_df["age"] + "_" + test_df["sex"]
    test_df["demo_encoded"] = test_df["demo_group"].astype('category').cat.codes
    test_df["accuracy"] = test_df.apply(lambda x: calculate_accuracy(x[pred_col], x[true_col]), axis=1) #10 different accuracy values observed from 0.28 to 1
    test_df["tpr"] = test_df.apply(lambda x: calculate_tpr(x[true_col], x[pred_col]), axis=1)
    test_df["fpr"] = test_df.apply(lambda x: calculate_fpr(x[true_col], x[pred_col]), axis=1)
    test_df["acc_quantile"] = pd.qcut(test_df["accuracy"], q=10, labels=False, duplicates = 'drop') # Returns the quantile labels and drop the number of quantiles if required
    test_df['tpr_quantile'] = pd.qcut(test_df["tpr"], q=10, labels=False, duplicates = 'drop')
    test_df['fpr_quantile'] = pd.qcut(test_df["fpr"], q=10, labels=False, duplicates = 'drop')
    test_df["match"] = (test_df[pred_col[0]] == test_df[true_col[0]]).astype(int)

    num_bins = test_df["acc_quantile"].nunique()
    print("Number of quantiles created:", num_bins)
    #test_df.to_csv('/home/pr2762@mc.cumc.columbia.edu/attributions_data.csv', index=False)

    variables = ['race', 'age', 'sex', 'demo_group']
    if disease_based:
        y_variables = ['match']
    else:
        y_variables = ['acc_quantile', 'tpr_quantile', 'fpr_quantile']
    
    #Create contingency table
    if not disease_based:
        for var in variables:
            contingency = pd.crosstab(test_df["acc_quantile"], test_df[var])
            proportions = contingency.div(contingency.sum(axis=1), axis=0)
            contingency.to_csv(f'csv_files/{var}_contingency_table.csv')
            proportions.to_csv(f'csv_files/{var}_proportion_table.csv')
        #Pearson Correlation Table
        quantiles = pd.get_dummies(test_df["acc_quantile"], prefix = 'quantile')
        quantile_labels = quantiles.columns
        for var in variables:
            demographics = pd.get_dummies(test_df[var])
            demographic_labels = demographics.columns
            correlation_df = pd.concat([quantiles, demographics], axis=1)
            correlation_matrix = correlation_df.corr(method='pearson')
            pearson_correlation = correlation_matrix.loc[quantile_labels, demographic_labels]
            pearson_correlation.to_csv(f'csv_files/{var}_correlation.csv')

    #Cramer's V Table
    test_df['acc_quantile'] = test_df['acc_quantile'].astype('category')
    test_df['tpr_quantile'] = test_df['tpr_quantile'].astype('category')
    test_df['fpr_quantile'] = test_df['fpr_quantile'].astype('category')
    results = {}
    printing = False
    if not disease_based:
        for y_var in y_variables:
            results[y_var] = {}
            for var in variables:
                results[y_var][var] = cramers_v(test_df[var], test_df[y_var], printing)
        results_flat = []
        for y_var, var_dict in results.items():
            for var, cramers_value in var_dict.items():
                results_flat.append({
                    'metric': y_var,
                    'demographic_variable': var,
                    'cramers_v': cramers_value
                })

        results_df = pd.DataFrame(results_flat)
        results_df.to_csv('csv_files/cramersV_results.csv', index=False)
    else:
        for y_var in y_variables:
            for var in variables:
                results[var] = cramers_v(test_df[var], test_df[y_var], printing)
        return results

    #Shapley Analysis
    if shapley:
        shapley_analysis_quantiles(test_df)

def bootstrapping_quantile_analysis(test_df, lowest_quantile, highest_quantile, lowest_quantile_idx, highest_quantile_idx):
    df = pd.DataFrame(columns = ['Group 1', 'Group 2', 'Group 1 Quantile', 'Group 2 Quantile', 'Accuracy: Group 1', 'Accuracy: Group 2', 
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
    print("95% CI for accuracy difference between highest and lowest quantile: ", lower, upper)
    p_one_sided, significance_one_sided = calculate_p_value(obs_score, scores, "accuracy", 'Group 1', 'Group 2')
    
    data = pd.DataFrame([{'Group 1': lowest_quantile,'Group 1 Quantile': lowest_quantile_idx, 'Group 2': highest_quantile, 'Group 2 Quantile': highest_quantile_idx, 'Accuracy: Group 1': x_observed, 'Accuracy: Group 2': y_observed,
    'P-One-Sided': p_one_sided, 'Significance-One-Sided': significance_one_sided}])
    df = pd.concat([data, df], ignore_index=True)
    return df

def cramers_v(x, y, printing=False):
    confusion_matrix = pd.crosstab(x, y)
    if printing:
        print(confusion_matrix)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def shapley_analysis_quantiles(df):
    X = pd.get_dummies(df[["race", "age", "sex"]], drop_first = False) #Use drop_first for linear models to avoid multicollinearity
    y = df["acc_quantile"]
    n_classes = y.nunique()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #Ensure train and test splits have the same distribution of y
    
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8, 
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    importance_df.to_csv('csv_files/FeatureImportance.csv')

    #Shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    per_class_importance = []
    for class_idx in range(n_classes):
        class_shap = shap_values[class_idx]
        importance = np.abs(class_shap).mean(axis=0)
        per_class_importance.append(
            pd.DataFrame({
                "feature": X.columns,
                "class": class_idx,
                "importance": importance
            })
        )
    importance_by_class = pd.concat(per_class_importance)
    importance_by_class.to_csv('csv_files/ImportanceByClass.csv')
