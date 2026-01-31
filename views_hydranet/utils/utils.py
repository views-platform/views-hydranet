import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import wandb
from torchvision import transforms

logger = logging.getLogger(__name__)

def _get_feature_indices(
    config: dict[str, Any], columns: list[str] | None = None
) -> tuple[int, int]:
    """
    Unified helper to determine feature start and end indices.
    Priority: 1. Column-name lookup, 2. Config key, 3. Hardcoded fallback (5).
    """
    if columns is not None:
        target_indicators = ["sb", "ns", "os"]
        feature_start_idx = -1
        for i, col in enumerate(columns):
            if any(ind in col.lower() for ind in target_indicators):
                feature_start_idx = i
                break
        if feature_start_idx == -1:
            feature_start_idx = config.get("first_feature_idx", 5)
    else:
        # Check config or default
        feature_start_idx = config.get("first_feature_idx", 5)

    requested_channels = config.get("input_channels", 3)
    return feature_start_idx, feature_start_idx + requested_channels


# networks
# learning rate schedulers
from torch.optim.lr_scheduler import (
    CyclicLR,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.utils.focal_loss import FocalLoss
from views_hydranet.utils.mtloss import MultiTaskLoss

# loss functions
from views_hydranet.utils.shringkage_loss import ShrinkageLoss
from views_hydranet.utils.warmup_decay_lr_scheduler import WarmupDecayLearningRateScheduler


def choose_model(config: dict, device: torch.device) -> nn.Module:
    """Chooses a model based on the provided configuration.

    This function acts as a factory for creating model instances. The model type
    is determined by the `config["model"]` string.

    Args:
        config: A dictionary containing model configuration, including:
            - "model" (str): The name of the model to instantiate.
            - Other keys required by the model's constructor
              (e.g., "input_channels", "dropout_rate").
        device: The PyTorch device to which the model should be moved.

    Returns:
        An instance of the chosen model, moved to the specified device.

    Raises:
        ValueError: If an unknown model name is provided in the config.
    """

    if config["model"] == "HydraBNUNet06_LSTM4":
        model = HydraBNUNet06_LSTM4(
            config["input_channels"],
            config["total_hidden_channels"],
            config["output_channels"],
            config["dropout_rate"],
        ).to(device)
    else:
        error_msg = f"Unknown model type: {config['model']}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return model


def choose_loss(config, device):
    """Selects loss functions based on the provided configuration.

    This function acts as a factory for creating loss function instances.
    It selects a regression loss and a classification loss based on the `config`
    dictionary.

    Args:
        config (dict): A dictionary containing loss function configuration, including:
                       - "loss_reg" (str): The name of the regression loss ('a' for MSE, 'b' for Shrinkage).
                       - "loss_class" (str): The name of the classification loss ('a' for BCE, 'b' for Focal).
                       - Other keys required by the loss constructors (e.g., "loss_reg_a").
        device (torch.device): The PyTorch device to which the loss functions should be moved.

    Returns:
        tuple: A tuple containing the regression criterion, the classification criterion,
               and an instance of the MultiTaskLoss.

    Raises:
        SystemExit: If an unknown `loss_reg` or `loss_class` name is provided in the config.
    """
    if config['loss_reg'] == 'a':
        criterion_reg = nn.MSELoss().to(device)

    elif config["loss_reg"] == "b":
        criterion_reg = ShrinkageLoss(
            a=config["loss_reg_a"], c=config["loss_reg_c"], size_average=True
        ).to(device)

    else:
        logger.error("Wrong reg loss...")
        sys.exit()
        return

    if config["loss_class"] == "a":
        criterion_class = nn.BCELoss().to(device)

    elif config["loss_class"] == "b":
        criterion_class = FocalLoss(
            alpha=config["loss_class_alpha"], gamma=config["loss_class_gamma"]
        ).to(device)  # THIS IS IN USE

    else:
        logger.error("Wrong class loss...")
        sys.exit()
        return

    logger.info(f"Regression loss: {criterion_reg}\n classification loss: {criterion_class}")

    is_regression = torch.Tensor(
        [True, True, True, False, False, False]
    )  # for vea you can just have 1 extre False (classifcation) in the end for the kl...
    multitaskloss_instance = MultiTaskLoss(
        is_regression, reduction="sum"
    )  # also try mean

    return (criterion_reg, criterion_class, multitaskloss_instance)


def choose_scheduler(config, unet):
    """Selects and configures a learning rate scheduler and optimizer.

    This function acts as a factory for creating a learning rate scheduler and its
    associated optimizer based on the `config` dictionary. It supports several
    scheduler types, including 'plateau', 'step', 'linear', 'CosineAnnealingLR',
    'OneCycleLR', 'CyclicLR', and 'WarmupDecay'.

    Args:
        config (dict): A dictionary containing scheduler configuration, including:
                       - "scheduler" (str): The name of the scheduler to use.
                       - "learning_rate" (float): The initial learning rate for the optimizer.
                       - Other keys required by the scheduler/optimizer constructors.
        unet (torch.nn.Module): The model whose parameters the optimizer will manage.

    Returns:
        tuple: A tuple containing the configured optimizer and scheduler. If the
               scheduler name in the config is not recognized, the scheduler
               will be an empty list.
    """
    if config["scheduler"] == "plateau":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = ReduceLROnPlateau(optimizer)

    elif config["scheduler"] == "step":  # seems to be an DEPRECATION issue
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = StepLR(optimizer, step_size=60)

    elif config["scheduler"] == "linear":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = LinearLR(optimizer)

    elif config["scheduler"] == "CosineAnnealingLR1":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        # you should try with config.samples * 0.2, 0,33 and 0.5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["samples"], eta_min=0.00005
        )

    elif config["scheduler"] == "CosineAnnealingLR02":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["samples"] * 0.2, eta_min=0.00005
        )

    elif config["scheduler"] == "CosineAnnealingLR033":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["samples"] * 0.33, eta_min=0.00005
        )

    elif config["scheduler"] == "CosineAnnealingLR05":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["samples"] * 0.5, eta_min=0.00005
        )

    elif config["scheduler"] == "CosineAnnealingLR004":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["samples"] * 0.04, eta_min=0.00005
        )

    elif config["scheduler"] == "OneCycleLR":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = OneCycleLR(
            optimizer,
            total_steps=32,
            max_lr=config["learning_rate"],
            anneal_strategy="cos",
        )

    elif config["scheduler"] == "CyclicLR":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        scheduler = CyclicLR(
            optimizer,
            step_size_up=200,
            base_lr=config["learning_rate"] * 0.1,
            max_lr=config["learning_rate"],
            mode="triangular2",
        )

    elif config["scheduler"] == "WarmupDecay":
        optimizer = torch.optim.AdamW(
            unet.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999)
        )
        # dimension of the input window
        d = config["window_dim"] * config["window_dim"] * config["input_channels"]
        scheduler = WarmupDecayLearningRateScheduler(
            optimizer, d=d, warmup_steps=config["warmup_steps"]
        )

    else:
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
        )
        scheduler = []  # could set to None...

    return (optimizer, scheduler)



def init_weights(m, config):
    """Initializes the weights of convolutional and linear layers within a module.

    This function applies a specified weight initialization method to `nn.Conv2d`
    and `nn.Linear` layers within the given module `m`. The initialization method
    is determined by the `config['weight_init']` parameter.

    Supported initialization methods:
    - 'xavier_uni': Xavier uniform initialization.
    - 'xavier_norm': Xavier normal initialization.
    - 'kaiming_uni': Kaiming uniform initialization (He initialization).
    - 'kaiming_norm': Kaiming normal initialization (He initialization).

    Args:
        m (torch.nn.Module): The module or layer whose weights are to be initialized.
        config (dict): A dictionary containing configuration, including:
                       - "weight_init" (str): The name of the weight initialization method.
    """
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
                       - "first_feature_idx" (int): The starting index of normalization features.
                       - "input_channels" (int): The number of features to normalize.
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

    first_feature_idx = config["first_feature_idx"]
    last_feature_idx = first_feature_idx + config["input_channels"] - 1

    for i in range(first_feature_idx, last_feature_idx + 1):
        feature = full_vol[:, :, :, i]

        feature_max = feature.max()
        feature_min = 0

        feature_norm = (b - a) * (feature - feature_min) / (feature_max - feature_min) + a

        full_vol[:, :, :, i] = feature_norm

    return full_vol


def get_data(config) -> np.ndarray:
    """Loads pre-processed data (Numpy volumes) based on the specified run_type.

    This function constructs a file path using the `model_path.data_processed` attribute
    and the `run_type` from the configuration. It then attempts to load a Numpy
    volume from this path.

    Args:
        config (dict): A dictionary containing configuration parameters, including:
                       - "run_type" (str): The partition to load (e.g., "calibration", "testing").

    Returns:
        np.ndarray: The loaded 4D Numpy array (`views_vol`).

    Raises:
        SystemExit: If the specified data file is not found.
    """

    # Data
    # Use path from config if available, else exit
    path_processed_str = config.get("path_processed_data")
    if not path_processed_str:
        logger.error("get_data: 'path_processed_data' not found in config. Exiting.")
        sys.exit()

    path_processed = Path(path_processed_str)

    run_type = config["run_type"]  # 'calibration', 'testing' or 'forecasting'

    try:
        file_name = f"{run_type}_vol.npy"
        # debug print
        logger.info(f"Loading {run_type} data from {file_name}...")
        views_vol = np.load(path_processed / file_name)

    except FileNotFoundError as e:
        logger.error(
            f"File not found: {e}. Run correct dataloader get_calibration_data.py, "
            f"get_test_data.py or get_forecasting_data.py. Now exiting..."
        )
        sys.exit()
        return

    return views_vol


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


def unit_norm(x, noise=False):
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

    if noise:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_unit_norm += (
            torch.randn(len(x_unit_norm), dtype=torch.float, requires_grad=False, device=device)
            * x_unit_norm.std()
        )

    return x_unit_norm


def standard(x, noise=False):
    """Standardize a 1D NumPy array by removing the mean and scaling to unit variance.

    Optionally adds Gaussian noise to the standardized array.

    Args:
        x (np.ndarray): The input 1D NumPy array to standardize.
        noise (bool, optional): If True, adds noise to the standardized array. Defaults to False.

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

    if noise:
        x_standard += np.random.normal(loc=0, scale=x_standard.std(), size=len(x_standard))

    return x_standard


def my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio):
    """Calculates a decayed number of events with a linear decay function.

    The decay function is defined by a `slope_ratio` and the total `samples`.
    The calculated value `y` is bounded by `min_events` (floor) and
    `roof_ratio * max_events` (roof).

    Args:
        sample (int): The current sample number, influencing the decay.
        samples (int): The total number of samples over which the decay occurs.
        min_events (int): The floor value for the number of events.
        max_events (int): The initial maximum number of events, from which decay starts.
        slope_ratio (float): A ratio that influences the steepness of the linear decay.
        roof_ratio (float): Upper bound ratio (0.0 to 1.0) applied to `max_events`.

    Returns:
        int: The calculated number of events `y`.

    Example:
        >>> # Normal decay case, respecting the roof
        >>> my_decay(sample=0, samples=100, min_events=10, max_events=100,
        ...          slope_ratio=1.0, roof_ratio=0.8)
        80
    """

    b = (-max_events + min_events) / (samples * slope_ratio)
    y = max_events + b * sample

    y = min(y, max_events * roof_ratio)
    y = max(y, min_events)

    return int(y)


def get_window_index(
    views_vol: np.ndarray, config: dict, sample: int, columns: list[str] | None = None
) -> dict:
    """Samples a spatial cell (row and column index) from the input `views_vol`."""

    # Use the unified helper to find features
    ln_best_sb_idx, last_feature_idx = _get_feature_indices(config, columns)

    min_events = config.get("min_events", 5)
    samples = config.get("samples", 300)
    slope_ratio = config.get("slope_ratio", 0.75)
    roof_ratio = config.get("roof_ratio", 0.7)

    # Identification of conflict heads
    fatcats = np.arange(ln_best_sb_idx, last_feature_idx, 1)
    n_fatcats = len(fatcats)

    fatcat = fatcats[sample % n_fatcats]
    views_vol_count = np.count_nonzero(views_vol[:, :, :, fatcat], axis=0)

    # --- Decay Logic ---
    max_events = views_vol_count.max()
    min_events = my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio)

    # number of events so >= 1 or > 0 is the same as np.nonzero
    min_events_index = np.where(views_vol_count >= min_events)

    min_events_row = min_events_index[0]
    min_events_col = min_events_index[1]

    # it is index... Not lat long.
    min_events_indx = [(row, col) for row, col in zip(min_events_row, min_events_col)]

    # indx = random.choice(min_events_indx) RANDOMENESS!!!!
    # dumb but working solution of np.random instead of random
    indx = min_events_indx[np.random.choice(len(min_events_indx))]

    # if you want a random temporal window, it is here.
    window_index = {"row_indx": indx[0], "col_indx": indx[1]}

    return window_index


def get_window_coords(window_index: dict, config: dict) -> dict:
    """Determines the spatial boundaries (coordinates) for a data window.

    Given a `window_index` (row and column of the anchor point) and a `window_dim`,
    this function calculates the boundaries. It uses `np.clip` to ensure
    coordinates stay within the `180x180` spatial dimensions.

    Args:
        window_index (dict): Anchor point with keys 'row_indx' and 'col_indx'.
        config (dict): Contains "window_dim" (int).

    Returns:
        dict: Calculated window coordinates:
              'min_row_indx', 'max_row_indx', 'min_col_indx', 'max_col_indx', and 'dim'.
    """

    window_dim = config["window_dim"]

    # Randomly select a window around the sampled index.
    min_row_indx = np.clip(
        int(window_index["row_indx"] - np.random.randint(0, window_dim)), 0, 180 - window_dim
    )
    max_row_indx = min_row_indx + window_dim
    min_col_indx = np.clip(
        int(window_index["col_indx"] - np.random.randint(0, window_dim)), 0, 180 - window_dim
    )
    max_col_indx = min_col_indx + window_dim

    # make dict of window coords to return
    window_coords = {
        "min_row_indx": min_row_indx,
        "max_row_indx": max_row_indx,
        "min_col_indx": min_col_indx,
        "max_col_indx": max_col_indx,
        "dim": window_dim,
    }

    return window_coords


def apply_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()


def train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list):
    avg_loss = np.mean(avg_loss_list)
    avg_loss_reg = np.mean(avg_loss_reg_list)
    avg_loss_class = np.mean(avg_loss_class_list)

    if wandb.run is not None:
        wandb.log(
            {"avg_loss": avg_loss, "avg_loss_reg": avg_loss_reg, "avg_loss_class": avg_loss_class}
        )


def get_train_tensors(
    views_vol: np.ndarray,
    sample: int,
    config: dict,
    device: str,
    columns: list[str] | None = None,
) -> torch.Tensor:
    """Creates a training tensor (spatial window) for a single training sample."""

    # 1. Determine time steps for hold-out
    time_steps = config.get("time_steps", 36)
    train_views_vol = views_vol if time_steps == 0 else views_vol[:-time_steps]

    # 2. Sample Window
    window_index = get_window_index(
        views_vol=views_vol, config=config, sample=sample, columns=columns
    )
    window_coords = get_window_coords(window_index=window_index, config=config)

    # 3. Extract Window
    input_window = train_views_vol[
        :,
        window_coords["min_row_indx"] : window_coords["max_row_indx"],
        window_coords["min_col_indx"] : window_coords["max_col_indx"],
        :,
    ]

    # 4. Slice and Permute
    ln_best_sb_idx, last_feature_idx = _get_feature_indices(config, columns)

    # JIT Scaling Logic: Ensure training data scale matches eval path
    from views_hydranet.utils.utils_scaling import ScalingEngine

    scaler = ScalingEngine.from_config(config)

    input_window_scaled = input_window.copy()
    # ALL features (channels 5+) are scaled here regardless of metadata
    # Metadata channels (0-4) are preserved as-is
    input_window_scaled[:, :, :, ln_best_sb_idx:last_feature_idx] = scaler.scale(
        input_window[:, :, :, ln_best_sb_idx:last_feature_idx], "get_train_tensors"
    )

    # Transform: [T, H, W, C] -> [1, T, C, H, W]
    train_tensor = (
        torch.tensor(input_window_scaled)
        .float()
        .to(device)
        .unsqueeze(dim=0)
        .permute(0, 1, 4, 2, 3)[:, :, ln_best_sb_idx:last_feature_idx, :, :]
    )

    # INTEGRITY CHECK: Log max feature value to detect explosion early
    if sample == 0 and train_tensor.numel() > 0:
        logger.info(
            f"AUDIT: Training Input Scale (Max Feature): {train_tensor.max().item():.4f}"
        )

    # 5. Apply Spatial Transforms (Random Flips)
    # We apply the same transform across all time steps to maintain consistency
    N, T, C, H, W = train_tensor.shape
    train_tensor_reshaped = train_tensor.reshape(N, T * C, H, W)

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    train_tensor_transformed = transform(train_tensor_reshaped)
    train_tensor = train_tensor_transformed.reshape(N, T, C, H, W)

    return train_tensor


def get_full_tensor(
    views_vol: np.ndarray, config: dict | None = None, columns: list[str] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts input 4D volume array into feature and metadata PyTorch tensors.

    Dynamically determines the split point between metadata (Identity) and
    features (Conflict data) based on provided column names.

    Args:
        views_vol: The 4D input volume [Time, Lat, Lon, Channels].
        config: Model configuration dictionary.
        columns: Optional list of column names in the order they appear in the channels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (full_tensor, metadata_tensor)
    """

    # 1. Identify Identity Columns
    # Identity aliases: priogrid_gid, pg_id, row, col, month_id, month, c_id

    # 2. Determine the Split Index
    if columns is not None:
        # Calculate index dynamically based on pattern-matching
        target_indicators = ["sb", "ns", "os"]
        feature_start_idx = -1

        for i, col in enumerate(columns):
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in target_indicators):
                feature_start_idx = i
                break

        if feature_start_idx == -1:
            logger.warning(
                f"get_full_tensor: Could not find any conflict targets in columns: {columns}. "
                f"Defaulting to index 5."
            )
            feature_start_idx = 5
        else:
            logger.debug(
                f"Dynamic Slicing: Found feature start at index {feature_start_idx} "
                f"('{columns[feature_start_idx]}')."
            )
    else:
        # Fallback to legacy behavior for backward compatibility with un-annotated volumes
        feature_start_idx = 5
        logger.warning("get_full_tensor: No columns provided. Falling back to hardcoded index 5.")

    # 3. Validation: The Handshake
    requested_channels = config["input_channels"] if config is not None else 3
    last_feature_idx = feature_start_idx + requested_channels
    actual_channels = views_vol.shape[-1]

    if actual_channels < last_feature_idx:
        raise ValueError(
            f"Architecture Mismatch! Model expects {requested_channels} features starting at "
            f"index {feature_start_idx}, but volume only has {actual_channels} total channels. "
            f"Please check QuerySet/Architecture alignment."
        )

    # 4. Perform Slicing and Scaling
    from views_hydranet.utils.utils_scaling import ScalingEngine

    scaler = ScalingEngine.from_config(config)

    # Transform to float array for JIT scaling (Polymorphic handle)
    if isinstance(views_vol, torch.Tensor):
        views_vol_np = views_vol.detach().cpu().numpy().astype(np.float32)
    else:
        views_vol_np = views_vol.copy().astype(np.float32)

    # UNCONDITIONAL: Scale all feature channels regardless of metadata
    views_vol_np[:, :, :, feature_start_idx:last_feature_idx] = scaler.scale(
        views_vol_np[:, :, :, feature_start_idx:last_feature_idx], "get_full_tensor"
    )

    vol_tensor = torch.tensor(views_vol_np).float()
    vol_tensor = vol_tensor.unsqueeze(dim=0).permute(0, 1, 4, 2, 3)

    full_tensor = vol_tensor[:, :, feature_start_idx:last_feature_idx, :, :]
    metadata_tensor = vol_tensor[:, :, :feature_start_idx, :, :]

    if full_tensor.numel() > 0:
        logger.info(f"AUDIT: Eval Input Scale (Max Feature): {full_tensor.max().item():.4f}")
    logger.debug(f"Slicing and scaling complete: full_tensor={full_tensor.shape}")

    return full_tensor, metadata_tensor


# Old implementation of get_full_tensor for reference.
# This version only returned a single tensor and did not separate metadata.
#
# def get_full_tensor(views_vol, config, device):
#
#     """
#     Uses to get the features for the full tensor
#     Used for out-of-sample predictions for both evaluation and forecasting, depending on the run_type (partition).
#     The test tensor is of size 1 x config.time_steps x config.input_channels x 180 x 180.
#     """
#
#     ln_best_sb_idx = config.first_feature_idx # 5 = ln_best_sb
#     last_feature_idx = ln_best_sb_idx + config.input_channels
#
#     print(f'views_vol shape {views_vol.shape}')  # (months, 180, 180, 8)
#
#     full_tensor = torch.tensor(views_vol).float().unsqueeze(dim=0).permute(0,1,4,2,3)[:, :, ln_best_sb_idx:last_feature_idx, :, :]
#
#     print(f'full_tensor shape {full_tensor.shape}') # (1, months, 3, 180, 180)
#
#     return full_tensor



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
    Potentially freezing the hidden state/short mem, the cell state/long mem, or both.
    Also have a random option where the model randomly picks between what to freeze.

    The function returns the new hidden state/short term memory h_tt and the prediction
    t1_pred and t1_pred_class.
    """

    if config["freeze_h"] == "hl":  # freeze the long term memory
        # split h_tt into hs_tt and hl_tt and save hl_tt as the forzen cell state/long term memory.
        # Call it hl_frozen. Half of the second dimension which is channels.
        split = int(h_tt.shape[1] / 2)
        _, hl_frozen = torch.split(h_tt, split, dim=1)
        t1_pred, t1_pred_class, h_tt = model(t0, h_tt)
        # Again split the h_tt into hs_tt and hl_tt. But discard the hl_tt
        hs, _ = torch.split(h_tt, split, dim=1)
        # Concatenate the frozen cell state/long term memory (hl_frozen) with the new
        # hidden state/short term memory. this is the new h_tt
        h_tt = torch.cat((hs, hl_frozen), dim=1)

    elif config["freeze_h"] == "hs":  # freeze the short term memory
        split = int(h_tt.shape[1] / 2)
        hs_frozen, _ = torch.split(h_tt, split, dim=1)
        t1_pred, t1_pred_class, h_tt = model(t0, h_tt)
        _, hl = torch.split(h_tt, split, dim=1)
        h_tt = torch.cat((hs_frozen, hl), dim=1)

    elif config["freeze_h"] == "all":  # freeze both h_l and h_s
        t1_pred, t1_pred_class, _ = model(t0, h_tt)

    elif config["freeze_h"] == "none":  # dont freeze
        t1_pred, t1_pred_class, h_tt = model(t0, h_tt)  # dont freeze anything.

    elif config["freeze_h"] == "random":  # random pick between what tho freeze
        t1_pred, t1_pred_class, h_tt_new = model(t0, h_tt)

        # splitting the tensor four ways along dim 1 to get hs1, hs2, hl1, and hl2
        split_four_ways = int(h_tt.shape[1] / 8)

        # split the h_tt from the last step
        (
            hs_1_frozen,
            hs_2_frozen,
            hs_3_frozen,
            hs_4_frozen,
            hl_1_frozen,
            hl_2_frozen,
            hl_3_frozen,
            hl_4_frozen,
        ) = torch.split(h_tt, split_four_ways, dim=1)

        # split the h_tt from the current step
        (
            hs_1_new,
            hs_2_new,
            hs_3_new,
            hs_4_new,
            hl_1_new,
            hl_2_new,
            hl_3_new,
            hl_4_new,
        ) = torch.split(h_tt_new, split_four_ways, dim=1)

        # make pairs of the frozen and new hidden states
        pairs = [
            (hs_1_frozen, hs_1_new),
            (hs_2_frozen, hs_2_new),
            (hs_3_frozen, hs_3_new),
            (hs_4_frozen, hs_4_new),
            (hl_1_frozen, hl_1_new),
            (hl_2_frozen, hl_2_new),
            (hl_3_frozen, hl_3_new),
            (hl_4_frozen, hl_4_new),
        ]
        # concatenate the frozen and new hidden states.
        # Randomly pick between the frozen and new hidden states for each pair.
        h_tt = torch.cat(
            [pair[0] if torch.rand(1) < 0.5 else pair[1] for pair in pairs], dim=1
        )

    else:
        logger.error("Wrong freeze option...")
        sys.exit()

    return t1_pred, t1_pred_class, h_tt


def weigh_loss(loss, y_t0, y_t1, distance_scale):
    """
    This function is used to weigh the loss function with a distance penalty.
    If the distance between y_t0 and y_t1 is large, i.e. the level of violence differ,
    then the loss is increased.
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
