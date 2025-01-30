import os

import numpy as np
import pickle
import time
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb


import sys
from pathlib import Path
from views_pipeline_core.managers.model import ModelPathManager


print("Current Working Directory:", os.getcwd())  # Check where Python is running from
print("Python Path:", sys.path)  # List of directories where Python looks for modules

# everything that has to do with eval will out. Some of this might also now be covered by Model manager or similar
from views_hydranet.utils.utils_prediction import sample_posterior
from views_hydranet.utils.utils_true_forecasting import make_forecast_storage_vol
from views_hydranet.utils.utils_hydranet_outputs import output_to_df, save_model_outputs
# from views_hydranet.utils.utils_hydranet_outputs import output_to_df, save_model_outputs, update_output_dict, retrieve_metadata, reshape_vols_to_arrays

from views_hydranet.utils.utils_model_outputs import ModelOutputs

#from utils_artifacts import get_latest_model_artifact # how to get this from model manager?

# SO VERY IMPORTANT THE THE VIEWS_VOL HERE DOES NOT NEED TO BE THE ONE FORCASTING PARTITION WE TRAINED ON
# We'll only retrain once a year after all
def forecast_posterior(model, views_vol, config, device):
#def forecast_posterior(df, config):

    """
    Retrive true forecasts form sample_posterior and generate comprehensive DataFrame and volume representations of these forecasts.

    This function handles posterior predictions for multiple features over time, calculates mean and standard deviations,
    and compiles these metrics along with metadata into a DataFrame suitable for evaluation or further analysis.
    Additionally, it constructs a volume for visualizations or plotting purposes.

    Args:
        model (torch.nn.Module): Trained model used to generate posterior predictions.
        df (pd.DataFrame): DataFrame containing the initial data for generating forecast storage volume (month_id, pg_id, c_id, col, row).
        config (dict): Configuration dictionary with model parameters, including 'time_steps' and 'month_range'. BAD TERMONOLOGY!!!!
        device (torch.device): Device to run model computations (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: 
            - DataFrame containing processed model outputs with scores, variances, and metadata.
            - 4D volume array suitable for testing and plotting.
    """

    # AS SOON AS YOU HAVE TRAINED A ARTIFACT YOU SHOULD USE THE FUNCTION BELOW TO GET THE POSTERIOR PREDICTIONS
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    posterior_list, posterior_list_class, _, _, _, _ = sample_posterior(model, views_vol, config, device)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # --------------------
    # for testing - you havn't trained a real forecast partition model yet.... There is prolly some code below you can use for that.
    #PATH_posterior_dict = "/home/simon/Documents/scripts/views_pipeline/models/purple_alien/data/generated/posterior_dict_36_calibration_20240613_165106.pkl"

    # get the posterior_dict from the pickle file in generated
    # with open(PATH_posterior_dict, 'rb') as f:
    #     posterior_dict = pickle.load(f)

    # month_range = 36 # MAGIC NUMBER ALERT - this is the number of months in the future we are forecasting
    # posterior_list, posterior_list_class, _ = posterior_dict['posterior_list'], posterior_dict['posterior_list_class'], posterior_dict['out_of_sample_vol'] 
    #----------------------



    month_range = config['time_steps']

    # storage volume for forecasts
    forecast_storage_vol = make_forecast_storage_vol(month_range = month_range, to_tensor=True) # CONFIG

    # Initialize dictionary to store outputs for different features
    dict_of_outputs_dicts = {k: ModelOutputs.make_output_dict(steps = month_range) for k in ["sb", "ns", "os"]} # CONFIG - step is month_range... BAD NAME!
 
    # Calculate mean and standard deviation for posterior predictions and class probabilities
    mean_array = np.array(posterior_list).mean(axis=0)
    std_array = np.array(posterior_list).std(axis=0)
    mean_class_array = np.array(posterior_list_class).mean(axis=0)
    std_class_array = np.array(posterior_list_class).std(axis=0)

    for t in range(mean_array.shape[0]):  # Iterate over time steps
        for i, j in enumerate(dict_of_outputs_dicts.keys()):  # Iterate over feature keys ('sb', 'ns', 'os')
            step = f"step{str(t + 1).zfill(2)}"

            # Reshape the arrays to 1D to create the dict of outputs.
            y_score, y_score_prob, y_var, y_var_prob = reshape_vols_to_arrays(t, i, mean_array, mean_class_array, std_array, std_class_array)

            # Retrieve metadata for the current time step
            pg_id, c_id, month_id = retrieve_metadata(t, forecast_storage_vol, forecast = True)

            # Update the output dictionary with the current predictions and metadata
            dict_of_outputs_dicts = update_output_dict(dict_of_outputs_dicts, t, j, step, y_score, y_score_prob, y_var, y_var_prob, pg_id, c_id, month_id)

    # Convert the output dictionaries to a DataFrame
    df_full = output_to_df(dict_of_outputs_dicts, forecast=True)

    # Drop columns related to observed data, as this is forecast-specific
    df_full = df_full.drop(columns=['y_true_sb', 'y_true_binary_sb', 'y_true_ns', 'y_true_binary_ns', 'y_true_os', 'y_true_binary_os'])

    # and make the vol just for testing and plotting - you should do this for eval as well. 
    vol_full = np.concatenate((mean_array, mean_class_array, forecast_storage_vol.squeeze().numpy()), axis=1)
    vol_full = np.transpose(vol_full, (0, 2, 3, 1))

  
    posterior_dict = {'posterior_list' : posterior_list, 'posterior_list_class': posterior_list_class, 'out_of_sample_vol' : None}
    #save_model_outputs(PATH, config, posterior_dict, dict_of_outputs_dicts)

    return df_full, vol_full, dict_of_outputs_dicts, posterior_dict




# ------------------------------------------------ ADAPT -----------------------------------------------------


def forecast_with_model_artifact(config, device, views_vol, PATH_ARTIFACTS, artifact_name=None):
#def handle_evaluation(config, device, views_vol, PATH_ARTIFACTS, artifact_name=None):

    """
    Loads a model artifact and use to to produce true forecasts using the correcet partition (Forecasting).

    This function handles the loading of a model artifact either by using a specified artifact name
    or by selecting the latest model artifact based on the run type (default). It then produced true forecasts 
    using the the model's posterior distribution and saves the output.

    Args:
        config: Configuration object containing parameters and settings.
        device: The device to run the model on (CPU or GPU).
        views_vol: The tensor containing the input data for evaluation.
        PATH_ARTIFACTS: The path where model artifacts are stored.
        artifact_name (optional): The specific name of the model artifact to load. Defaults to None.

    Raises:
        FileNotFoundError: If the specified or default model artifact cannot be found.

    """

    # if an artifact name is provided through the CLI, use it. Otherwise, get the latest model artifact based on the run type
    if artifact_name:
        print(f"Using (non-default) artifact: {artifact_name}")
        
        # If the pytorch artifact lacks the file extension, add it. This is obviously specific to pytorch artifacts, but we are deep in the model code here, so it is fine.
        if not artifact_name.endswith('.pt'):
            artifact_name += '.pt'
        
        # Define the full (model specific) path for the artifact
        #PATH_MODEL_ARTIFACT = os.path.join(PATH_ARTIFACTS, artifact_name)

        # pathlib alternative as per sara's comment
        PATH_MODEL_ARTIFACT = PATH_ARTIFACTS / artifact_name # PATH_ARTIFACTS is already a Path object
    
    else:
        # use the latest model artifact based on the run type
        print(f"Using latest (default) run type ({config.run_type}) specific artifact")
        
        # Get the latest model artifact based on the run type and the (models specific) artifacts path
        PATH_MODEL_ARTIFACT = get_latest_model_artifact(PATH_ARTIFACTS, config.run_type)

    # Check if the model artifact exists - if not, raise an error
    #if not os.path.exists(PATH_MODEL_ARTIFACT):
    #    raise FileNotFoundError(f"Model artifact not found at {PATH_MODEL_ARTIFACT}")
    
    # Pathlib alternative as per sara's comment
    if not PATH_MODEL_ARTIFACT.exists(): # PATH_MODEL_ARTIFACT is already a Path object
        raise FileNotFoundError(f"Model artifact not found at {PATH_MODEL_ARTIFACT}")

    # load the model
    model = torch.load(PATH_MODEL_ARTIFACT)
    
    # get the exact model date_time stamp for the pkl files made in the evaluate_posterior from evaluation.py
    #model_time_stamp = os.path.basename(PATH_MODEL_ARTIFACT)[-18:-3] # 18 is the length of the timestamp string + ".pt", and -3 is to remove the .pt file extension. a bit hardcoded, but very simple and should not change.


    # Pathlib alternative as per sara's comment
    model_time_stamp = PATH_MODEL_ARTIFACT.stem[-15:] # 15 is the length of the timestamp string. This is more robust than the os.path.basename solution above since it does not rely on the file extension.

    # print for debugging
    print(f"model_time_stamp: {model_time_stamp}")

    # add to config for logging and conciseness
    config.model_time_stamp = model_time_stamp

    # evaluate the model posterior distribution
    df_forecast, vol_forecast, dict_of_outputs_dicts, posterior_dict = forecast_posterior(model, views_vol, config, device)
    
    # So a bit wierd, but df_forecast and df_eval are both created in the save_model_outputs function.... 
    # This is how it works in the eval fuction, but I think I like it better like here where the df is created in the forecast (/eval) function.
    # .... align alter 
    save_model_outputs(PATH, config, posterior_dict, dict_of_outputs_dicts, forecast_vol = vol_forecast, forecast = True)


    #save_model_outputs(PATH, config, posterior_dict, dict_of_outputs_dicts, dict_of_eval_dicts = None, forecast_vol = None, full_tensor = None, metadata_tensor = None):

    # done. 
    print('Done forecasting') 

