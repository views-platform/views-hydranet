import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from views_pipeline_core.managers.model import ModelPathManager

from views_hydranet.deprecated.metrics import EvaluationMetrics
from views_hydranet.utils.utils_model_outputs import ModelOutputs

logger = logging.getLogger(__name__)

def predictions_to_contract_df(
    posterior_list: List[np.ndarray],
    forecast_storage_vol: np.ndarray,
    target: str,
) -> List[pd.DataFrame]:
    """
    Converts raw posterior samples into the list of DataFrames required by views-evaluation.

    This function implements the "Producer Contract" for views-hydranet.
    It takes a list of posterior samples (from sample_posterior) and maps them to
    their corresponding month_id and location_id (from forecast_storage_vol).

    Args:
        posterior_list: List of N posterior samples, each with shape [steps, features, H, W].
        forecast_storage_vol: Metadata volume with shape [batch, steps, features, H, W].
                              Channel 0: priogrid_gid, 3: month_id, 4: c_id.
        target: The target variable name (e.g., 'sb').

    Returns:
        List[pd.DataFrame]: A list of DataFrames, one per sequence (usually 1 here).
                            Each DF has MultiIndex (month_id, priogrid_gid)
                            and column f"pred_lr_{target}".
    """

    # We assume batch size 1 for metadata extraction as per Hydranet convention
    # forecast_storage_vol shape: [1, steps, 8, 180, 180]
    steps = forecast_storage_vol.shape[1]
    
    # Extract IDs and flatten
    pg_ids = forecast_storage_vol[0, :, 0, :, :].reshape(steps, -1)
    month_ids = forecast_storage_vol[0, :, 3, :, :].reshape(steps, -1)
    c_ids = forecast_storage_vol[0, :, 4, :, :].reshape(steps, -1)

    # Filter out ocean cells (pg_id == 0)
    # We create a mask for valid land cells across all steps
    mask = pg_ids > 0
    
    # posterior_list is List of [steps, features, H, W]
    # We need to stack samples to get [samples, steps, features, H, W]
    all_samples = np.stack(posterior_list)  # [S, T, F, H, W]
    
    # Mapping target string to index
    target_map = {"sb": 0, "ns": 1, "os": 2}
    t_idx = target_map.get(target, 0)
    
    # Extract target feature samples: [S, T, H, W]
    target_samples = all_samples[:, :, t_idx, :, :]
    
    # Re-scale/Inverse transform: exp(x) - 1 for 'ln_' prefixed models
    # Hydranet targets are usually log-transformed.
    # The contract requires RAW COUNTS.
    target_samples = np.exp(target_samples) - 1
    
    # Now we build the DataFrame for this sequence
    # Since we are usually dealing with one predictive run at a time in this module,
    # the list will contain one DataFrame.
    
    rows = []
    # This is slightly inefficient but clear. For 180x180 it's fine.
    # We iterate over steps and land-cells.
    for t in range(steps):
        valid_pg = pg_ids[t][mask[t]]
        valid_months = month_ids[t][mask[t]]
        # target_samples[:, t, :, :] is [S, H, W] -> flatten to [S, H*W] -> filter land: [S, Valid]
        valid_samples = target_samples[:, t, :, :].reshape(len(posterior_list), -1)[:, mask[t]]
        
        for i in range(len(valid_pg)):
            rows.append({
                "month_id": int(valid_months[i]),
                "priogrid_gid": int(valid_pg[i]),
                f"pred_lr_{target}": valid_samples[:, i].tolist()
            })
            
    df = pd.DataFrame(rows)
    df = df.set_index(["month_id", "priogrid_gid"])
    
    return [df]

def zstack_to_contract_df(
    posterior_zstack: np.ndarray,
    meta_zstack: np.ndarray,
    target: str,
) -> List[pd.DataFrame]:
    """
    Converts zstacks from HydraNetInference into the list of DataFrames required by views-evaluation.

    Args:
        posterior_zstack: [steps, H, W, channels, samples]. 
                          Channels 0,1,2 are magnitudes (sb, ns, os).
        meta_zstack: [steps, H, W, channels, 1].
                     Channel 0: priogrid_gid, 3: month_id.
        target: The target variable name (e.g., 'sb').

    Returns:
        List[pd.DataFrame]: Contract-compliant predictions.
    """
    steps = posterior_zstack.shape[0]
    samples = posterior_zstack.shape[-1]
    
    # Mapping target to channel index
    target_map = {"sb": 0, "ns": 1, "os": 2}
    t_idx = target_map.get(target, 0)
    
    # Extract magnitudes and inverse transform
    # [steps, H, W, samples]
    mags = posterior_zstack[:, :, :, t_idx, :]
    mags = np.exp(mags) - 1
    
    # Extract IDs
    pg_ids = meta_zstack[:, :, :, 0, 0]
    month_ids = meta_zstack[:, :, :, 3, 0]
    
    mask = pg_ids > 0
    
    rows = []
    for t in range(steps):
        valid_pg = pg_ids[t][mask[t]]
        valid_months = month_ids[t][mask[t]]
        # [H, W, S] -> flatten spatial -> [H*W, S] -> filter: [Valid, S]
        valid_mags = mags[t].reshape(-1, samples)[mask[t].flatten()]
        
        for i in range(len(valid_pg)):
            rows.append({
                "month_id": int(valid_months[i]),
                "priogrid_gid": int(valid_pg[i]),
                f"pred_lr_{target}": valid_mags[i].tolist()
            })
            
    df = pd.DataFrame(rows)
    df = df.set_index(["month_id", "priogrid_gid"])
    return [df]

def validate_contract_dataframes(list_df: List[pd.DataFrame]) -> None:
    """
    Validates that the contract DataFrames are robust and safe for evaluation.
    
    Checks for:
    1. Non-finite values (NaN, Inf) in predictions.
    2. Presence of ocean cells (priogrid_gid == 0).
    3. Empty DataFrames.

    Raises:
        ValueError: If any validation rule is violated.
    """
    if not list_df:
        raise ValueError("Contract DataFrame list is empty!")

    for i, df in enumerate(list_df):
        if df.empty:
            raise ValueError(f"Sequence {i} is empty!")
            
        # Check for Ocean Cells in Index
        pg_ids = df.index.get_level_values("priogrid_gid")
        if (pg_ids == 0).any():
            raise ValueError(f"Sequence {i} contains ocean cells (priogrid_gid=0)!")

        # Check for Non-Finite Numbers in all columns
        # Contract columns contain lists, so we need to flatten to check
        for col in df.columns:
            # Flatten lists of samples to a single array for fast checking
            all_values = np.concatenate(df[col].values)
            if not np.isfinite(all_values).all():
                num_bad = (~np.isfinite(all_values)).sum()
                raise ValueError(
                    f"Sequence {i}, column {col} contains {num_bad} non-finite values (NaN/Inf)!"
                )

    logger.info("Adversarial data validation passed: Data is finite and land-only.")

def contract_df_to_zstack(
    list_df_predictions: List[pd.DataFrame],
    meta_zstack: np.ndarray,
    target: str,
) -> np.ndarray:
    """
    Inverse operation of zstack_to_contract_df. 
    Reconstructs the posterior_zstack magnitudes from the Contract DataFrame.

    This function is the "Reversibility Proof". It proves we can recover 
    the original identical model outputs from the flattened contract format.

    Args:
        list_df_predictions: The list of contract DataFrames.
        meta_zstack: The spatial template [steps, H, W, channels, 1].
        target: The target variable name.

    Returns:
        np.ndarray: Reconstructed magnitudes [steps, H, W, 1, samples].
                    (Note: returns single target channel).
    """
    df = list_df_predictions[0]
    steps, H, W, _, _ = meta_zstack.shape
    
    # Peek at first list to get sample count
    samples = len(df.iloc[0][f"pred_lr_{target}"])
    
    # Pre-allocate reconstructed volume
    reconstructed = np.zeros((steps, H, W, 1, samples))
    
    # Extract IDs from template
    pg_ids_template = meta_zstack[:, :, :, 0, 0]
    month_ids_template = meta_zstack[:, :, :, 3, 0]
    
    # Inverse transform column name
    col = f"pred_lr_{target}"
    
    # Iterate over template steps/cells
    for t in range(steps):
        # We find all entries in the DF for this month
        month_id = int(np.unique(month_ids_template[t])[0])
        df_month = df.xs(month_id, level="month_id")
        
        for h in range(H):
            for w in range(W):
                pg_id = int(pg_ids_template[t, h, w])
                if pg_id > 0:
                    # Retrieve samples from DF
                    raw_samples = df_month.loc[pg_id, col]
                    # Apply log transform: ln(x + 1) to reverse exp(x) - 1
                    reconstructed[t, h, w, 0, :] = np.log(np.array(raw_samples) + 1)
                else:
                    # Ocean is 0 in original log-space too
                    reconstructed[t, h, w, 0, :] = 0.0
                    
    return reconstructed

def output_to_df(dict_of_outputs_dicts):
    
    """
    Converts the dictionary of model outputs into a consolidated pandas DataFrame, formatted for HydraNet.

    This function takes dictionaries of model outputs for different target variables ('sb', 'ns', 'os'), 
    converts them into separate DataFrames, renames columns to distinguish between different targets, 
    and then merges them into a single DataFrame. The merged DataFrame excludes ocean cells (where `c_id == 0`).

    Args:
        dict_of_outputs_dicts (dict): A dictionary containing sub-dictionaries of model outputs 
                                      for different targets ('sb', 'ns', 'os'). Each sub-dictionary 
                                      should be structured with keys as steps and values as `ModelOutputs` instances.

    Returns:
        df_all (pd.DataFrame): A DataFrame where columns from different targets are suffixed with '0', '1', or '2' 
                      respectively. The DataFrame is cleaned to exclude ocean cells and has columns 
                      properly typed for use in HydraNet.

    Example:
        >>> dict_of_outputs_dicts = {
                'sb': {'step01': ModelOutputs(...), ...},
                'ns': {'step01': ModelOutputs(...), ...},
                'os': {'step01': ModelOutputs(...), ...}
            }
        >>> df_full = output_to_df(dict_of_outputs_dicts)
        >>> print(df_full.head())
    """

     # Example usage with 'sb', 'ns', 'os'
    df_sb = ModelOutputs.output_dict_to_dataframe(dict_of_outputs_dicts["sb"])
    df_ns = ModelOutputs.output_dict_to_dataframe(dict_of_outputs_dicts["ns"])
    df_os = ModelOutputs.output_dict_to_dataframe(dict_of_outputs_dicts["os"])

    # SO FROM HERE IT GETS VERY HydraNet SPECIFIC. 
    common_cols = ["priogrid_gid", "c_id", "month_id", "step"] # D: KeyError: "['step'] not found in axis"

    # rename the columns so that the onse in df_test2 ends with a 0 and the ones in df_test3 ends with a 1. don't change the common columns
    # df_sb.columns = [f"{i}_sb" if i not in common_cols else i for i in df_sb.columns]
    # df_ns.columns = [f"{i}_ns" if i not in common_cols else i for i in df_ns.columns]
    # df_os.columns = [f"{i}_os" if i not in common_cols else i for i in df_os.columns]
    for col in common_cols:
        if col in df_sb.columns:
            df_sb = df_sb.drop(columns=[col])
        if col in df_ns.columns:
            df_ns = df_ns.drop(columns=[col])
        if col in df_os.columns:
            df_os = df_os.drop(columns=[col])

    # drop the priogrid_gid and c_id columns from df_ns and df_os - bc concat is faster than merge when they are sorted the same way.
    # df_sb = df_sb.drop(columns=common_cols)
    # df_ns = df_ns.drop(columns=common_cols)


    # merge the dataframes
    df_all = pd.concat([df_sb, df_ns, df_os], axis=1)

    # drop ocean cells, i.e. where c_id == 0
    # Check if 'c_id' column exists before filtering
    if 'c_id' in df_all.columns:
        df_all = df_all[df_all["c_id"] != 0]
    else:
        logger.warning("Column 'c_id' not found in the DataFrame")


    # no you can just drop it
    df_all = df_all.reset_index(drop=True)

    # change all columns to float
    df_all = df_all.astype(float)

    # make the binary columns integers
    # df_all = df_all.astype({"y_true_binary_sb": int, "y_true_binary_ns": int, "y_true_binary_os": int, "month_id" : int, "step" : int})

    # Check if columns exist before changing their data types
    columns_to_int = ["y_true_binary_sb", "y_true_binary_ns", "y_true_binary_os", "month_id", "step"]
    for col in columns_to_int:
        if col in df_all.columns:
            df_all[col] = df_all[col].astype(int)
        else:
            logger.warning(f"Column '{col}' not found in the DataFrame")

    # print the df
    #df_all

    return df_all


def evaluation_to_df(dict_of_eval_dicts):

    """
    Converts a dictionary of evaluation metric dictionaries into a consolidated DataFrame.

    This function takes a dictionary containing evaluation metric dictionaries for different features
    ('sb', 'ns', 'os'), converts each evaluation dictionary to a DataFrame, renames the columns to 
    reflect their respective feature, and merges them into a single DataFrame.

    Args:
        dict_of_eval_dicts (Dict[str, Dict[str, EvaluationMetrics]]): A dictionary where keys are feature 
            identifiers ('sb', 'ns', 'os') and values are dictionaries of evaluation metrics per time step 
            for each feature. Each evaluation metric dictionary is expected to contain instances of `EvaluationMetrics`.

    Returns:
        pd.DataFrame: A consolidated DataFrame containing evaluation metrics for all features. The DataFrame 
        includes columns for each metric with suffixes '_sb', '_ns', and '_os' to denote the feature they belong to.
        Each row corresponds to a specific time step across the different features.

    Example:
        >>> dict_of_eval_dicts = {
        ...     'sb': {'step01': EvaluationMetrics(MSE=0.1, AP=0.2, AUC=0.3, Brier=0.4), ...},
        ...     'ns': {'step01': EvaluationMetrics(MSE=0.5, AP=0.6, AUC=0.7, Brier=0.8), ...},
        ...     'os': {'step01': EvaluationMetrics(MSE=0.9, AP=1.0, AUC=1.1, Brier=1.2), ...}
        ... }
        >>> df_all_eval = evaluation_to_df(dict_of_eval_dicts)
        >>> print(df_all_eval.head())
           MSE_sb  AP_sb  AUC_sb  Brier_sb  ...  MSE_os  AP_os  AUC_os  Brier_os
        0     0.1    0.2     0.3       0.4  ...     0.9    1.0     1.1       1.2
        ...
    """

    df_sb_eval = EvaluationMetrics.evaluation_dict_to_dataframe(dict_of_eval_dicts['sb'])
    df_ns_eval = EvaluationMetrics.evaluation_dict_to_dataframe(dict_of_eval_dicts['ns'])
    df_os_eval = EvaluationMetrics.evaluation_dict_to_dataframe(dict_of_eval_dicts['os'])

    df_sb_eval.columns = [f"{i}_sb" for i in df_os_eval.columns]
    df_ns_eval.columns = [f"{i}_ns" for i in df_os_eval.columns]
    df_os_eval.columns = [f"{i}_os" for i in df_os_eval.columns]

    # merge the dataframes
    df_all_eval = pd.concat([df_sb_eval, df_ns_eval, df_os_eval], axis=1)

    return df_all_eval


def save_model_outputs(model_path: ModelPathManager, config, posterior_dict, dict_of_outputs_dicts, dict_of_eval_dicts, full_tensor, metadata_tensor):
    """
    Sets up data paths, creates necessary directories, and saves model outputs including posterior dictionary, 
    evaluation metrics, and tensors to pickle files.

    Args:
        PATH (str): The base path for saving data.
        config (object): Configuration object containing attributes such as time_steps, run_type, and model_time_stamp.
        posterior_dict (dict): Dictionary containing posterior list, posterior list class, and out-of-sample volume.
        dict_of_outputs_dicts (dict): Dictionary containing model outputs.
        dict_of_eval_dicts (dict): Dictionary containing evaluation metrics.
        full_tensor (torch.Tensor): Tensor containing full dataset for predictions.
        metadata_tensor (torch.Tensor): Tensor containing metadata for the dataset.
    """

    # Create the directory if it does not exist
    Path(model_path.data_generated).mkdir(parents=True, exist_ok=True)
    print(f'PATH to generated data: {model_path.data_generated}')

    # Convert dicts of outputs and evaluation metrics to DataFrames
    df_sb_os_ns_output = output_to_df(dict_of_outputs_dicts)
    df_sb_os_ns_evaluation = evaluation_to_df(dict_of_eval_dicts)

    # Save the posterior dictionary
    posterior_path = f'{model_path.data_generated}/posterior_dict_{config["time_steps"]}_{config["run_type"]}_{config["model_time_stamp"]}.pkl'
    with open(posterior_path, 'wb') as file:
        pickle.dump(posterior_dict, file)

    # Save the DataFrame of model outputs
    outputs_path = f'{model_path.data_generated}/df_sb_os_ns_output_{config["time_steps"]}_{config["run_type"]}_{config["model_time_stamp"]}.pkl'
    with open(outputs_path, 'wb') as file:
        pickle.dump(df_sb_os_ns_output, file)

    # Save the DataFrame of evaluation metrics
    evaluation_path = f'{model_path.data_generated}/df_sb_os_ns_evaluation_{config["time_steps"]}_{config["run_type"]}_{config["model_time_stamp"]}.pkl'
    with open(evaluation_path, 'wb') as file:
        pickle.dump(df_sb_os_ns_evaluation, file)

    # Save the tensors
    test_vol_path = f'{model_path.data_generated}/test_vol_{config["time_steps"]}_{config["run_type"]}_{config["model_time_stamp"]}.pkl'
    with open(test_vol_path, 'wb') as file:
        pickle.dump(full_tensor.cpu().numpy(), file)

    metadata_vol_path = f'{model_path.data_generated}/metadata_vol_{config["time_steps"]}_{config["run_type"]}_{config["model_time_stamp"]}.pkl'
    with open(metadata_vol_path, 'wb') as file:
        pickle.dump(metadata_tensor.cpu().numpy(), file)

    print('Posterior dict, outputs, evaluation metrics, and tensors pickled and saved!')


# Why have you place this in here?

def plot_metrics(df_all, feature = 0):

    """
    Plots MSE, Average Precision, ROC AUC, and Brier Score for each month from log_dict_list.

    Args:
        log_dict_list (list of dict): List of dictionaries with monthly metrics.
        num_months (int): Number of months to plot.
    """

    # Initialize lists to store metrics for each month
    mse_list = []
    ap_list = []
    auc_list = []
    brier_list = []

    df_all["month"] = df_all["step"] #super quick fix, super lazy


    # Iterate over the log_dict_list and extract the metrics
    for i in df_all["month"].unique():

        y_score = df_all[df_all["month"] == i][f"y_score_{feature}"]
        y_score_prob = df_all[df_all["month"] == i][f"y_score_prob_{feature}"]
        y_true = df_all[df_all["month"] == i][f"y_true_{feature}"]
        y_true_binary = df_all[df_all["month"] == i][f"y_true_binary_{feature}"]

        mse = mean_squared_error(y_true, y_score)
        ap = average_precision_score(y_true_binary, y_score_prob)
        auc = roc_auc_score(y_true_binary, y_score_prob)
        brier = brier_score_loss(y_true_binary, y_score_prob)


        mse_list.append(mse)
        ap_list.append(ap)
        auc_list.append(auc)
        brier_list.append(brier)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # Plot MSE
    axs[0, 0].plot(range(1, len(mse_list) + 1), mse_list, marker='o', color='b', label='MSE')
    axs[0, 0].set_title('Mean Squared Error')
    axs[0, 0].set_xlabel('Month')
    axs[0, 0].set_ylabel('MSE')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Average Precision
    axs[0, 1].plot(range(1, len(ap_list) + 1), ap_list, marker='o', color='g', label='Average Precision')
    axs[0, 1].set_title('Average Precision Score')
    axs[0, 1].set_xlabel('Month')
    axs[0, 1].set_ylabel('AP Score')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot ROC AUC
    axs[1, 0].plot(range(1, len(auc_list) + 1), auc_list, marker='o', color='r', label='ROC AUC')
    axs[1, 0].set_title('ROC AUC Score')
    axs[1, 0].set_xlabel('Month')
    axs[1, 0].set_ylabel('AUC Score')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Brier Score
    axs[1, 1].plot(range(1, len(brier_list) + 1), brier_list, marker='o', color='m', label='Brier Score')
    axs[1, 1].set_title('Brier Score Loss')
    axs[1, 1].set_xlabel('Month')
    axs[1, 1].set_ylabel('Brier Score')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # add a title
    plt.suptitle(f'Metrics for Feature {feature} Over {df_all["month"].max()} Months', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Show plots
    plt.show()