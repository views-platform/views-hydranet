import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, average_precision_score, roc_auc_score, brier_score_loss

import wandb

import sys
from pathlib import Path
from views_hydranet.utils.utils_hydranet_outputs import evaluation_to_df

# there are things in other utils that should be here...

def add_wandb_monthly_metrics():

    """
    Defines the WandB metrics for monthly evaluation.

    This function sets up the metrics for logging monthly evaluation metrics in WandB.
    It defines a step metric called "monthly/out_sample_month" and specifies that any 
    metric under the "monthly" namespace will use "monthly/out_sample_month" as its step metric.

    Usage:
        This function should be called at the start of a WandB run to configure 
        how metrics are tracked over time steps.

    Example:
        >>> wandb.init(project="example_project")
        >>> add_wandb_monthly_metrics()
        >>> wandb.log({"monthly/mean_squared_error": 0.02, "monthly/out_sample_month": 1})

    Notes:
        - The step metric "monthly/out_sample_month" will be used to log metrics for each time  (i.e. forecasted month).
        - Any metric prefixed with "monthly/" will follow the "monthly/out_sample_month" step metric.

    See Also:
        - `wandb.define_metric`: WandB API for defining metrics and their step relationships.
    """
        
    # Define "new" monthly metrics for WandB logging
    wandb.define_metric("monthly/out_sample_month")
    wandb.define_metric("monthly/*", step_metric="monthly/out_sample_month")



def generate_wandb_log_dict(log_dict, dict_of_eval_dicts, feature, step):

    """
    Adds evaluation metrics to a WandB log dictionary for a specific feature and time step (i.e. forcasted month).

    This function updates the provided log dictionary with evaluation metrics from 
    a specified feature and step, formatted for WandB logging. It appends the metrics
    to the log dictionary using the "monthly/{metric_name}_{feature}" format.

    Args:
        log_dict (dict): The log dictionary to be updated with new metrics.
        dict_of_eval_dicts (Dict[str, Dict[str, EvaluationMetrics]]): A dictionary of evaluation metrics,
            where the keys are feature identifiers and the values are dictionaries with time steps as keys 
            and `EvaluationMetrics` instances as values.
        feature (str): The feature for which the metrics should be logged (e.g., 'sb', 'ns', 'os').
        step (str): The specific time step (month forecasted) for which metrics are logged (e.g., 'step01').

    Returns:
        dict: The updated log dictionary with the evaluation metrics for the specified feature and step.

    Example:
        >>> log_dict = {}
        >>> dict_of_eval_dicts = {
        ...     'sb': {'step01': EvaluationMetrics(MSE=0.1, AP=0.2, AUC=0.3, Brier=0.4), ...},
        ...     'ns': {'step01': EvaluationMetrics(MSE=0.5, AP=0.6, AUC=0.7, Brier=0.8), ...},
        ...     'os': {'step01': EvaluationMetrics(MSE=0.9, AP=1.0, AUC=1.1, Brier=1.2), ...}
        ... }
        >>> log_dict = generate_wandb_log_dict(log_dict, dict_of_eval_dicts, 'sb', 'step01')
        >>> print(log_dict)
        {
            'monthly/MSE_sb': 0.1,
            'monthly/AP_sb': 0.2,
            'monthly/AUC_sb': 0.3,
            'monthly/Brier_sb': 0.4
        }

    Notes:
        - Only non-None values from the `EvaluationMetrics` instance are added to the log dictionary.
        - The metrics are formatted with the "monthly/{metric_name}_{feature}" naming convention for WandB logging.

    See Also:
        - `wandb.log`: WandB API for logging metrics.
    """
    
    for key, value in dict_of_eval_dicts[feature][step].__dict__.items():
        if value is not None:
            log_dict[f"monthly/{key}_{feature}"] = value

    return log_dict 


def generate_wandb_mean_metrics_log_dict(dict_of_eval_dicts):
    """
    Calculates the mean of each evaluation metric from a dictionary of evaluation results and 
    returns a dictionary formatted for WandB logging.

    Args:
        dict_of_eval_dicts (Dict[str, Any]): Dictionary containing evaluation metrics for each step 
                                             and feature.

    Returns:
        Dict[str, float]: Dictionary with the mean value of each metric, formatted for WandB logging.

     Notes:
        Calculating the mean metrics across the month-specific metric values does not give the 
        same result as calculating the metrics on the full array of predictions at once.        
    """

    # Convert the dictionary of evaluation metrics to a DataFrame
    df_eval = evaluation_to_df(dict_of_eval_dicts)

    # Initialize a dictionary to store mean values of each metric
    mean_metric_log_dict = {}

    # Iterate through the columns (metrics) of the DataFrame
    for metric_name in df_eval.columns:
        mean_value = df_eval[metric_name].mean()
        # Check if the mean is a valid number (not NaN)
        if not pd.isna(mean_value):
            mean_metric_log_dict[metric_name] = mean_value

    return mean_metric_log_dict



# ---------------- Im unsure if this is used or not... ----------------

# 
def log_wandb_mean_metrics(config, df_eval):
    """
    Logs evaluation metrics to WandB.

    This function computes the mean of provided metrics and logs them to WandB for different targets.
    The metrics include mean squared error, average precision score, ROC AUC score, and Brier score loss.

    Args:
        config : Configuration object containing parameters and settings.
        df_eval (pd.DataFrame): DataFrame containing evaluation metrics with columns 
                                ['mean_squared_error_sb', 'average_precision_score_sb', 
                                 'roc_auc_score_sb', 'brier_score_loss_sb', 
                                 'mean_squared_error_ns', 'average_precision_score_ns', 
                                 'roc_auc_score_ns', 'brier_score_loss_ns', 
                                 'mean_squared_error_os', 'average_precision_score_os', 
                                 'roc_auc_score_os', 'brier_score_loss_os'].
    """
    # Drop NaN values before computing means
    metric_means = df_eval.mean(axis=0, skipna=True)

    metrics = ['mean_squared_error', 'average_precision_score', 'roc_auc_score', 'brier_score_loss']
    targets = ['sb', 'ns', 'os']

    # Log metrics for each target
    for target in targets:
        for metric in metrics:
            metric_name = f"{config.time_steps}month_{metric}_{target}"
            metric_value = metric_means.get(f"{metric}_{target}")
            #if pd.notna(metric_value):
            wandb.log({metric_name: metric_value})

    print(f"Logged metrics to WandB for targets: {targets}")
