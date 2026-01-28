import sys
from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss

import wandb
from views_pipeline_core.managers.model import ModelPathManager

# networks
from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

# loss functions
from views_hydranet.utils.shringkage_loss import ShrinkageLoss 
from views_hydranet.utils.focal_loss import FocalLoss 
from views_hydranet.utils.mtloss import MultiTaskLoss 

# learning rate schedulers
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LinearLR, OneCycleLR, CyclicLR
from views_hydranet.utils.warmup_decay_lr_scheduler import WarmupDecayLearningRateScheduler


def choose_model(config, device):
    """Chooses a model based on the provided configuration.

    This function acts as a factory for creating model instances. The model type
    is determined by the `config["model"]` string.

    Args:
        config (dict): A dictionary containing model configuration, including:
                       - "model" (str): The name of the model to instantiate.
                       - Other keys required by the model's constructor
                         (e.g., "input_channels", "dropout_rate").
        device (torch.device): The PyTorch device to which the model should be moved.

    Returns:
        torch.nn.Module: An instance of the chosen model, moved to the specified device.

    .. warning::
        If an unknown model name is provided in the config, this function will print
        "no model..." to stdout and then raise an `UnboundLocalError` because the
        `unet` variable is never assigned.
    """

    if config["model"] == 'HydraBNUNet06_LSTM4':
        unet = HydraBNUNet06_LSTM4(config["input_channels"], config["total_hidden_channels"], config["output_channels"], config["dropout_rate"]).to(device)

    else:
        print('no model...')

    return unet


def choose_loss(config, device):
    if config['loss_reg'] == 'a':
        criterion_reg = nn.MSELoss().to(device)

    elif config['loss_reg'] == 'b': 
        criterion_reg = ShrinkageLoss(a=config['loss_reg_a'], c=config['loss_reg_c'], size_average = True).to(device)

    else:
        print('Wrong reg loss...')
        sys.exit()

    if config['loss_class'] == 'a':
        criterion_class = nn.BCELoss().to(device)

    elif config['loss_class'] == 'b': 
        criterion_class =  FocalLoss(alpha = config['loss_class_alpha'], gamma=config['loss_class_gamma']).to(device) # THIS IS IN USE

    else:
        print('Wrong class loss...')
        sys.exit()

    print(f'Regression loss: {criterion_reg}\n classification loss: {criterion_class}')

    is_regression = torch.Tensor([True, True, True, False, False, False])   # for vea you can just have 1 extre False (classifcation) in the end for the kl... Or should it really be seen as a reg?
    multitaskloss_instance = MultiTaskLoss(is_regression, reduction = 'sum') # also try mean

    return(criterion_reg, criterion_class, multitaskloss_instance)


def choose_sheduler(config, unet):
    if config['scheduler'] == 'plateau':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer)

    elif config['scheduler'] == 'step': # seems to be an DEPRECATION issue
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = StepLR(optimizer, step_size= 60)

    elif config['scheduler'] == 'linear':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = LinearLR(optimizer)

    elif config['scheduler'] == 'CosineAnnealingLR1':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['samples'], eta_min = 0.00005) # you should try with config.samples * 0.2, 0,33 and 0.5

    elif config['scheduler'] == 'CosineAnnealingLR02':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['samples'] * 0.2, eta_min = 0.00005)

    elif config['scheduler'] == 'CosineAnnealingLR033':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['samples'] * 0.33, eta_min = 0.00005)

    elif config['scheduler'] == 'CosineAnnealingLR05':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['samples'] * 0.5, eta_min = 0.00005)

    elif config['scheduler'] == 'CosineAnnealingLR004':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['samples'] * 0.04, eta_min = 0.00005)


    elif config['scheduler'] == 'OneCycleLR':
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = OneCycleLR(optimizer,
                       total_steps=32, 
                       max_lr = config["learning_rate"], # Upper learning rate boundaries in the cycle for each parameter group
                       anneal_strategy = 'cos') # Specifies the annealing strategy

    elif config['scheduler'] == 'CyclicLR':

        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        scheduler = CyclicLR(optimizer,
                       step_size_up=200,
                       base_lr = config["learning_rate"] * 0.1,
                       max_lr = config["learning_rate"], # Upper learning rate boundaries in the cycle for each parameter group
                       mode = 'triangular2') # Specifies the annealing strategy
        
    elif config['scheduler'] == 'WarmupDecay':
        
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], betas = (0.9, 0.999))
        d = config['window_dim'] * config['window_dim'] * config['input_channels'] # this is the dimension of the input window
        scheduler = WarmupDecayLearningRateScheduler(optimizer, d = d, warmup_steps = config['warmup_steps'])


    else:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'], weight_decay = config['weight_decay'], betas = (0.9, 0.999))
        scheduler = [] # could set to None...

    return(optimizer, scheduler)



def init_weights(m, config):
    if config['weight_init'] == 'xavier_uni':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    elif config['weight_init'] == 'xavier_norm':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    elif config['weight_init'] == 'kaiming_uni':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            
    elif config['weight_init'] == 'kaiming_norm':
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    else:
        pass


def norm_features(full_vol: np.ndarray, config: dict, a: int = 0, b: int = 1) -> np.ndarray:
    """Normalizes a slice of features in a 4D volume in-place to the range [a, b].

    This function iterates through a range of features defined by `config['first_feature_idx']`
    and `config['input_channels']`. For each feature slice, it normalizes the values.

    Args:
        full_vol (np.ndarray): The 4D numpy array to be modified, with shape
                               [time, height, width, features].
        config (dict): A dictionary containing configuration, including:
                       - "first_feature_idx" (int): The starting index of the features to normalize.
                       - "input_channels" (int): The number of features to normalize.
                       - "un_log" (bool, optional): If True, applies `np.exp(feature) - 1` before
                         normalization. Defaults to False.
        a (int, optional): The lower bound of the normalization range. Defaults to 0.
        b (int, optional): The upper bound of the normalization range. Defaults to 1.

    Returns:
        np.ndarray: The same `full_vol` array, modified in-place.

    .. warning::
        - This function modifies the `full_vol` array in-place.
        - The minimum value for normalization is hardcoded to `0`. It does not use the
          actual minimum of the feature data.
        - If the maximum value of a feature slice is 0, this function will cause a
          `RuntimeWarning: invalid value encountered in divide` due to division by zero,
          resulting in `NaN` values in that feature slice.
    """

    first_feature_idx = config['first_feature_idx'] #config.first_feature_idx
    last_feature_idx = first_feature_idx + config['input_channels'] - 1 #config.first_feature_idx + config.input_channels - 1


    for i in range(first_feature_idx, last_feature_idx + 1):

        feature = full_vol[:, :, :, i] 

        if "un_log" in config and config["un_log"]:
            feature = np.exp(feature) - 1

        feature_max = feature.max() # could make sure that we are not using information from the future.... But this is not a big deal... 
        feature_min = 0 #full_vol[:, :, :, i].min()

        feature_norm = (b-a)*(feature - feature_min)/(feature_max-feature_min)+a

        full_vol[:,:,:,i] = feature_norm

    return full_vol


def get_data(config):

    """Return the data for either the calibration, the test run or an actual forecast.
    The shape for the views_vol is (N, C, H, W, D) where D is features.
    Right now the features are ln_best_sb, ln_best_ns, ln_best_os
    """

    # Data
    #location = config.path_processed_data
    # _, PATH_PROCESSED, _ = setup_data_paths(PATH)
    PATH_PROCESSED = model_path.data_processed

    run_type = config["run_type"] # 'calibration', 'testing' or 'forecasting'

    try:
        file_name = f'/{run_type}_vol.npy' # NOT WINDOWS FRIENDLY
        # debug print
        print(f'Loading {run_type} data from {file_name}...')
        views_vol = np.load(str(PATH_PROCESSED) + file_name)
    
    except FileNotFoundError as e:
        print(f'File not found: {e}. Run correct dataloader get_calibration_data.py, get_test_data.py or get_forecasting_data.py. Now exiting...')
        sys.exit()

    return(views_vol)


def norm(x, a = 0, b = 1):

    """Normalize a 1D array or Tensor to a specified range [a, b].

    By default, normalizes the input `x` to the range [0, 1].

    Args:
        x (np.ndarray or torch.Tensor): The input array or tensor to normalize.
        a (float, optional): The minimum value of the target range. Defaults to 0.
        b (float, optional): The maximum value of the target range. Defaults to 1.

    Returns:
        np.ndarray or torch.Tensor: The normalized array or tensor within the range [a, b].

    Example:
        >>> import numpy as np
        >>> norm(np.array([1, 2, 3, 4, 5]))
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> norm(np.array([10, 20, 30]), a=-1, b=1)
        array([-1.,  0.,  1.])
    """
    x_norm = (b-a)*(x - x.min())/(x.max()-x.min())+a
    return(x_norm)


def unit_norm(x, noise = False):

    """Normalizes a 1D PyTorch Tensor to a unit vector.

    Optionally adds Gaussian noise to the normalized vector.

    Args:
        x (torch.Tensor): The input 1D PyTorch Tensor to normalize.
        noise (bool, optional): If True, adds Gaussian noise to the unit vector. Defaults to False.

    Returns:
        torch.Tensor: The normalized 1D unit vector, optionally with added noise.

    Example:
        >>> import torch
        >>> # Example without noise
        >>> x_in = torch.tensor([3.0, 4.0])
        >>> unit_norm(x_in)
        tensor([0.6000, 0.8000])
        >>> # Example with noise (output will vary due to randomness)
        >>> x_in_noisy = torch.tensor([1.0, 1.0])
        >>> unit_norm(x_in_noisy, noise=True) # doctest: +SKIP
    """
    x_unit_norm = x / torch.linalg.norm(x)

    if noise == True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_unit_norm += torch.randn(len(x_unit_norm), dtype=torch.float, requires_grad=False, device = device) * x_unit_norm.std()

    return(x_unit_norm)


def standard(x, noise = False):

    """Standardize a 1D NumPy array by removing the mean and scaling to unit variance.

    Optionally adds Gaussian noise to the standardized array.

    Args:
        x (np.ndarray): The input 1D NumPy array to standardize.
        noise (bool, optional): If True, adds Gaussian noise to the standardized array. Defaults to False.

    Returns:
        np.ndarray: The standardized 1D NumPy array, optionally with added noise.

    Example:
        >>> import numpy as np
        >>> # Example without noise
        >>> x_in = np.array([1, 2, 3, 4, 5])
        >>> standard(x_in)
        array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
        >>> # Example with noise (output will vary due to randomness)
        >>> x_in_noisy = np.array([10, 20, 30])
        >>> standard(x_in_noisy, noise=True) # doctest: +SKIP
    """

    x_standard = (x - x.mean()) / x.std()

    if noise == True:
        x_standard += np.random.normal(loc = 0, scale = x_standard.std(), size = len(x_standard))

    return(x_standard)


def my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio):

    """Calculates a decayed number of events with a linear decay function, constrained by a floor and a roof.

    The decay function is defined by a `slope_ratio` and the total `samples`.
    The calculated value `y` is bounded by `min_events` (floor) and `roof_ratio * max_events` (roof).

    Args:
        sample (int): The current sample number, influencing the decay.
        samples (int): The total number of samples over which the decay occurs.
        min_events (int): The floor value for the number of events. The result `y` will not go below this.
        max_events (int): The initial maximum number of events, from which decay starts.
        slope_ratio (float): A ratio that influences the steepness of the linear decay.
        roof_ratio (float): A ratio (0.0 to 1.0) applied to `max_events` to set the upper bound (roof) for `y`.

    Returns:
        int: The calculated number of events `y`, constrained by `min_events` and `roof_ratio * max_events`.

    Example:
        >>> # Normal decay case, respecting the roof
        >>> my_decay(sample=0, samples=100, min_events=10, max_events=100, slope_ratio=1.0, roof_ratio=0.8)
        80
        >>> # Case where the decay hits the floor (min_events)
        >>> my_decay(sample=99, samples=100, min_events=10, max_events=100, slope_ratio=1.0, roof_ratio=1.0)
        10
        >>> # Case where the value remains at max_events (due to small slope_ratio)
        >>> my_decay(sample=0, samples=100, min_events=10, max_events=100, slope_ratio=0.1, roof_ratio=1.0)
        100
    """

    b = ((-max_events + min_events)/(samples*slope_ratio))
    y = (max_events + b * sample)
    
    y = min(y, max_events*roof_ratio)
    y = max(y, min_events)
    
    return(int(y))


def get_window_index(views_vol: np.ndarray, config: dict, sample: int) -> dict: 

    """Draw/sample a cell which serves as the ancor for the sampeled window/patch drawn from the traning tensor.
    The dimensions of the windows are HxWxD, 
    where H=D in {16,32,64} and D is the number of months in the training data.
    The windows are constrained to be sampled from an area with some
    minimum number of log_best events (min_events)."""


    # BY NOW THIS IS PRETTY HACKY... SHOULD BE MADE MORE ELEGANT AT SOME POINT..


    ln_best_sb_idx = config['first_feature_idx'] # 5 = ln_best_sb 
    last_feature_idx = ln_best_sb_idx + config['input_channels'] # removed -1 here. Now this is the last feature index.
    min_events = config['min_events']
    samples = config['samples']
    slope_ratio = config['slope_ratio']
    roof_ratio = config['roof_ratio']

    # NEW----------------------------------------------------------------------------------------------------------------------------
    fatcats = np.arange(ln_best_sb_idx, last_feature_idx, 1)
    n_fatcats = len(fatcats)

    fatcat = fatcats[sample % n_fatcats]
    views_vol_count = np.count_nonzero(views_vol[:,:,:,fatcat], axis = 0) #.sum(axis=2) #for either sb, ns, os
    
    # --------------------------------------------------------------------------------------------------------------------------------

    max_events = views_vol_count.max()
    min_events = my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio)
    
    min_events_index = np.where(views_vol_count >= min_events) # number of events so >= 1 or > 0 is the same as np.nonzero

    min_events_row = min_events_index[0]
    min_events_col = min_events_index[1]

    # it is index... Not lat long.
    min_events_indx = [(row, col) for row, col in zip(min_events_row, min_events_col)] 

    #indx = random.choice(min_events_indx) RANDOMENESS!!!!
    indx = min_events_indx[np.random.choice(len(min_events_indx))] # dumb but working solution of np.random instead of random

    # if you want a random temporal window, it is here.
    window_index = {'row_indx':indx[0], 'col_indx':indx[1]} 

    return(window_index)


def get_window_coords(window_index: dict, config: dict) -> dict:
    """Return the coordinates of the window around the sampled index. 
    This implementaions ensures that the window does never go out of bounds.
    (Thus no need for sampling until a window is found that does not go out of bounds)."""

    # you can change this back to random if you want
    window_dim = config["window_dim"]

    # Randomly select a window around the sampled index. np.clip is used to ensure that the window does not go out of bounds
    min_row_indx = np.clip(int(window_index['row_indx'] - np.random.randint(0, window_dim)), 0, 180 - window_dim)
    max_row_indx = min_row_indx + window_dim
    min_col_indx = np.clip(int(window_index['col_indx'] - np.random.randint(0, window_dim)), 0, 180 - window_dim)
    max_col_indx = min_col_indx + window_dim

    # make dict of window coords to return
    window_coords = {
        'min_row_indx':min_row_indx, 
        'max_row_indx':max_row_indx, 
        'min_col_indx':min_col_indx, 
        'max_col_indx':max_col_indx, 
        'dim':window_dim}

    return(window_coords)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list):

    avg_loss = np.mean(avg_loss_list)
    avg_loss_reg = np.mean(avg_loss_reg_list)
    avg_loss_class = np.mean(avg_loss_class_list)
    
    # also log maps...
    wandb.log({"avg_loss": avg_loss, "avg_loss_reg": avg_loss_reg, "avg_loss_class": avg_loss_class})


# Should rename to sub_tensor or something like that... But it is used for training.. 
def get_train_tensors(views_vol: np.ndarray, sample: int, config: dict, device: torch.device) -> torch.Tensor: 
    """
    Samples a window from the training tensor based on a sampled index and coordinates.

    This function prepares a training tensor by:
    1. Slicing the input `views_vol` to exclude future time steps for evaluation.
    2. Calling `get_window_index` to select a base spatial cell.
    3. Calling `get_window_coords` to determine the window boundaries around the selected cell.
    4. Extracting the `input_window` from `train_views_vol` based on these coordinates.
    5. Converting the `input_window` to a PyTorch tensor, moving it to the specified `device`,
       adding a batch dimension (`unsqueeze`), permuting dimensions, and slicing features
       based on `config["first_feature_idx"]` and `config["input_channels"]`.
    6. Applying spatial data augmentation (random horizontal and vertical flips) to the
       reshaped tensor.

    Args:
        views_vol (np.ndarray): A 4D numpy array with shape [n_months, height, width, n_features]
                                representing the full volume data.
        sample (int): The current sample number, used by `get_window_index` for sampling logic.
        config (dict): A dictionary containing configuration parameters such as:
                       - "time_steps" (int): Number of months to exclude from the end of `views_vol`.
                       - "first_feature_idx" (int): Starting index for the main features in `views_vol`.
                       - "input_channels" (int): Number of features to select for the `train_tensor`.
                       - "window_dim" (int): Dimension of the square spatial window.
        device (torch.device): The PyTorch device ('cuda' or 'cpu') to place the resulting tensor on.

    Returns:
        torch.Tensor: A 5D PyTorch tensor with shape
                      [1, (n_months - time_steps), input_channels, window_dim, window_dim]
                      ready for model training.

    .. warning::
        - This function relies on `get_window_index` and `get_window_coords`, both of which
          internally use `np.random.choice` and `np.random.randint`, making the window
          selection non-deterministic unless these functions are mocked or `np.random.seed()`
          is set globally.
        - The `torchvision.transforms` also introduce randomness if not explicitly controlled.
        - The comment "Should rename to sub_tensor or something like that... But it is used for training.."
          suggests a potential naming inconsistency that could lead to confusion.
    """

    # Not using the last 36 months - these ar for test set
    train_views_vol = views_vol if config["time_steps"] == 0 else views_vol[:-config["time_steps"]] # horrible naming... 

 #   min_max_values = 
    window_index = get_window_index(views_vol = views_vol, config = config, sample = sample) # you should try and take this out of the loop - so you keep the index but changes the window_coords!!!
    window_coords = get_window_coords(window_index = window_index, config = config)

    # you can add positional encoding here if you want - perhaps.....

    input_window = train_views_vol[ : , window_coords['min_row_indx'] : window_coords['max_row_indx'] , window_coords['min_col_indx'] : window_coords['max_col_indx'], :]

    ln_best_sb_idx = config["first_feature_idx"] # 5 = ln_best_sb
    last_feature_idx = ln_best_sb_idx + config["input_channels"]
    train_tensor = torch.tensor(input_window).float().to(device).unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, ln_best_sb_idx:last_feature_idx, :, :]

    # Reshape
    N = train_tensor.shape[0] # batch size. Always one - remember your do batch a different way here
    C = train_tensor.shape[1] # months
    D = config["input_channels"] # features
    H = train_tensor.shape[3] # height
    W =  train_tensor.shape[4] # width

    # add spatial transformer
    transformer = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])

    # data augmentation (can be turned of for final experiments)
    train_tensor = train_tensor.reshape(N, C*D, H, W)
    train_tensor = transformer(train_tensor[:,:,:,:])
    train_tensor = train_tensor.reshape(N, C, D, H, W)


    return train_tensor


#def get_full_tensor(views_vol, config, device):
#
#    """
#    Uses to get the features for the full tensor
#    Used for out-of-sample predictions for both evaluation and forecasting, depending on the run_type (partition). 
#    The test tensor is of size 1 x config.time_steps x config.input_channels x 180 x 180.
#    """
#
#    ln_best_sb_idx = config.first_feature_idx # 5 = ln_best_sb
#    last_feature_idx = ln_best_sb_idx + config.input_channels
#
#    print(f'views_vol shape {views_vol.shape}')  # (months, 180, 180, 8)
#
#    full_tensor = torch.tensor(views_vol).float().unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, ln_best_sb_idx:last_feature_idx, :, :] 
#
#    print(f'full_tensor shape {full_tensor.shape}') # (1, months, 3, 180, 180) 
#
#    return full_tensor 


def get_full_tensor(views_vol: np.ndarray, config: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts the input 4D volume array into PyTorch tensors for model input, separating feature and metadata tensors.

    This function transforms the input `views_vol`, which is a 4D numpy array, into two PyTorch tensors: 
    `full_tensor` and `metadata_tensor`. The `full_tensor` is used for model input and contains the features 
    specified for out-of-sample predictions. The `metadata_tensor` retains other columns not used as features 
    and is kept for metadata purposes.

    Args:
        views_vol (np.ndarray): A 4D numpy array with shape [n_months, height, width, n_features]. This volume 
                                contains spatial-temporal data with both features and metadata.
        config (Optional[dict]): A dictionary containing model configuration, particularly "input_channels".
                                 If `None`, "input_channels" defaults to 3 (hardcoded for testing purposes).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - `full_tensor` (torch.Tensor): A tensor with the selected features, of shape 
              [1, n_months, selected_features, height, width]. This tensor is used for out-of-sample 
              predictions for both evaluation and forecasting.
            - `metadata_tensor` (torch.Tensor): A tensor with the metadata, of shape 
              [1, n_months, metadata_features, height, width]. This tensor contains the remaining columns 
              of the original volume not used in `full_tensor`.

    Raises:
        ValueError: If the input `views_vol` does not have 4 dimensions.

    .. warning::
        - The `ln_best_sb_idx` (start index for main features) is currently hardcoded to `5`. This assumes 
          a fixed order and number of initial metadata features in `views_vol`.
        - When `config` is `None`, `input_channels` defaults to `3`. This behavior is hardcoded for 
          testing purposes and might not reflect production configuration.
        - The `print` statements to stdout are side effects.

    Example:
        >>> views_vol = np.random.rand(36, 180, 180, 8)  # Example input volume
        >>> # Example with config
        >>> config_with_channels = {"input_channels": 3}
        >>> full_tensor, metadata_tensor = get_full_tensor(views_vol, config_with_channels)
        >>> print(full_tensor.shape)
        torch.Size([1, 36, 3, 180, 180]) 
        >>> print(metadata_tensor.shape)
        torch.Size([1, 36, 5, 180, 180])
        >>> # Example with config=None
        >>> full_tensor_none_config, metadata_tensor_none_config = get_full_tensor(views_vol, config=None)
        >>> print(full_tensor_none_config.shape)
        torch.Size([1, 36, 3, 180, 180]) 
        >>> print(metadata_tensor_none_config.shape)
        torch.Size([1, 36, 5, 180, 180])
    """

    ln_best_sb_idx = 5#config.first_feature_idx # 5 = ln_best_sb

    if config is not None:
        last_feature_idx = ln_best_sb_idx + config["input_channels"]

    else:
        last_feature_idx = ln_best_sb_idx + 3 # hard coded, but just for testing when no config is passed

    print(f'views_vol shape {views_vol.shape}')

    # THIS IS WHERE YOU LOOSE THE OTHE FEATURES!!!!
    full_tensor = torch.tensor(views_vol).float().unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, ln_best_sb_idx:last_feature_idx, :, :] 

    # Make a metadata tensor with evrything else
    metadata_tensor = torch.tensor(views_vol).float().unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, :ln_best_sb_idx, :, :]

    print(f'full_tensor shape {full_tensor.shape}')

    return full_tensor, metadata_tensor



# def get_log_dict(i, mean_array, mean_class_array, std_array, std_class_array, out_of_sample_vol, config):
# 
#     """Return a dictionary of metrics for the monthly out-of-sample predictions for W&B."""
# 
#     log_dict = {}
#     log_dict["monthly/out_sample_month"] = i
# 
# 
#     #Fix in a sec when you see if it runs at all.... 
#     for j in range(3): #(config.targets): # TARGETS IS & BUT THIS SHOULD BE 3!!!!!
# 
#         y_score = mean_array[i,j,:,:].reshape(-1) # make it 1d  # nu 180x180 
#         y_score_prob = mean_class_array[i,j,:,:].reshape(-1) # nu 180x180 
#         
#         # do not really know what to do with these yet.
#         y_var = std_array[i,j,:,:].reshape(-1)  # nu 180x180  
#         y_var_prob = std_class_array[i,j,:,:].reshape(-1)  # nu 180x180 
# 
#         y_true = out_of_sample_vol[:,i,j,:,:].reshape(-1)  # nu 180x180 . dim 0 is time
#         y_true_binary = (y_true > 0) * 1
# 
# 
#         mse = mean_squared_error(y_true, y_score)
#         ap = average_precision_score(y_true_binary, y_score_prob)
#         auc = roc_auc_score(y_true_binary, y_score_prob)
#         brier = brier_score_loss(y_true_binary, y_score_prob)
# 
#         log_dict[f"monthly/mean_squared_error{j}"] = mse
#         log_dict[f"monthly/average_precision_score{j}"] = ap
#         log_dict[f"monthly/roc_auc_score{j}"] = auc
#         log_dict[f"monthly/brier_score_loss{j}"] = brier
# 
#     return log_dict
# 

def execute_freeze_h_option(config, model, t0, h_tt):

    """
    This function is used to execute the freeze option set in config.
    Potantially freezing the hidden state/short mem, the cell state/long mem, or both.
    Also have a random option where the model randomly picks between what to freeze.

    The function returns the new hidden state/short term memory h_tt and the prediction t1_pred and t1_pred_class.    
    """
     
    if config["freeze_h"] == "hl": # freeze the long term memory
        
        split = int(h_tt.shape[1]/2) # split h_tt into hs_tt and hl_tt and save hl_tt as the forzen cell state/long term memory. Call it hl_frozen. Half of the second dimension which is channels.
        _, hl_frozen = torch.split(h_tt, split, dim=1)
        t1_pred, t1_pred_class, h_tt = model(t0, h_tt) 
        hs, _ = torch.split(h_tt, split, dim=1) # Again split the h_tt into hs_tt and hl_tt. But discard the hl_tt
        h_tt = torch.cat((hs, hl_frozen), dim=1) # Concatenate the frozen cell state/long term memory (hl_frozen) with the new hidden state/short term memory. this is the new h_tt

    elif config["freeze_h"] == "hs": # freeze the short term memory

        split = int(h_tt.shape[1]/2) 
        hs_frozen, _ = torch.split(h_tt, split, dim=1)
        t1_pred, t1_pred_class, h_tt = model(t0, h_tt)
        _, hl = torch.split(h_tt, split, dim=1)
        h_tt = torch.cat((hs_frozen, hl), dim=1) 

    elif config["freeze_h"] == "all": # freeze both h_l and h_s

        t1_pred, t1_pred_class, _ = model(t0, h_tt) 

    elif config["freeze_h"] == "none": # dont freeze
        t1_pred, t1_pred_class, h_tt = model(t0, h_tt) # dont freeze anything.

    elif config["freeze_h"] == "random": # random pick between what tho freeze of hs1, hs2, hl1, and hl2

        t1_pred, t1_pred_class, h_tt_new = model(t0, h_tt)

        split_four_ways = int(h_tt.shape[1] / 8) # spltting the tensor four ways along dim 1 to get hs1, hs2, hl1, and hl2

        hs_1_frozen, hs_2_frozen, hs_3_frozen, hs_4_frozen, hl_1_frozen, hl_2_frozen, hl_3_frozen, hl_4_frozen = torch.split(h_tt, split_four_ways, dim=1) # split the h_tt from the last step
        hs_1_new, hs_2_new, hs_3_new, hs_4_new, hl_1_new, hl_2_new, hl_3_new, hl_4_new = torch.split(h_tt_new, split_four_ways, dim=1) # split the h_tt from the current step

        pairs = [(hs_1_frozen, hs_1_new), (hs_2_frozen, hs_2_new), (hs_3_frozen, hs_3_new), (hs_4_frozen, hs_4_new), (hl_1_frozen, hl_1_new), (hl_2_frozen, hl_2_new), (hl_3_frozen, hl_3_new), (hl_4_frozen, hl_4_new)] # make pairs of the frozen and new hidden states
        h_tt = torch.cat([pair[0] if torch.rand(1) < 0.5 else pair[1] for pair in pairs], dim=1) # concatenate the frozen and new hidden states. Randomly pick between the frozen and new hidden states for each pair.

    else:
        print('Wrong freez option...')
        sys.exit()

    return t1_pred, t1_pred_class, h_tt


def weigh_loss(loss, y_t0, y_t1, distance_scale):

    """
    This function is used to weigh the loss function with a distance penalty. 
    If the distance between y_t0 and y_t1 is large, i.e. the level of violence differ, then the loss is increased.
    The point is to make the model more sensitive to large changes in violence compared to inertia.
    """

    # Calculate the squared distance between y_t0 and y_t1
    squared_distance = torch.pow(y_t1 - y_t0, 2)
    
    # Add the distance penalty to the original loss
    new_loss = loss + torch.mean(squared_distance) * distance_scale

    return new_loss


# Define a custom learning rate function
# def custom_lr_lambda(step, warmup_steps, d)

#     """
#     Return a custom learning rate for the optimizer.
#     The learning rate is a function of the step number and the warmup_steps.
#     From the paper: attention is all you need.
#     """

#     return (d**(-0.5)) * min(step**(-0.5), step * warmup_steps**(-1.5))