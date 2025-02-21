import sys
import numpy as np  
import torch
import logging
from typing import Dict, Optional, List, Tuple
from torch.nn import Module

from views_hydranet.utils.utils import  get_full_tensor # rename when refactoring


class HydraNetInference:
    """
    Handles inference with the HydraNet model, including model loading,
    inference execution, and posterior sampling.
    """

    def __init__(self, model: Module, config: Dict, device: Optional[str] = None):
        """
        Initializes the inference pipeline for HydraNet.

        Args:
        - model (torch.nn.Module): The trained PyTorch model for inference. (Required)
        - config (dict): Configuration settings for inference.
        - device (str, optional): The device to run inference on ('cuda' or 'cpu').
                                  If not specified, it is automatically detected.
        """
        # Step 1: Determine the best available device
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logging.info(f"Using device: {self.device}")

        # Step 2: Validate inputs
        if not isinstance(model, Module):
            raise TypeError("Expected 'model' to be an instance of torch.nn.Module.")
        if not isinstance(config, dict):
            raise TypeError("Expected 'config' to be a dictionary.")

        self.model = model
        self.config = config

        # Step 3: Move model to device and configure for inference
        self.model.to(self.device)
        self.model.eval()
        self.model.apply(self._apply_dropout)

        logging.info("HydraNetInference initialized successfully.")


    def _apply_dropout(self, module: torch.nn.Module):
        """ Applies dropout during inference for approximate Bayesian uncertainty estimation. """
        if isinstance(module, torch.nn.Dropout):
            module.train()


    def execute_freeze_h_option(self, t0: torch.Tensor, h_tt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles the freezing of hidden state (`h_tt`) based on the configuration.

        This function selectively freezes short-term (`hs`) or long-term (`hl`) memory,
        or both, based on the `config["freeze_h"]` setting.

        Args:
        - t0 (torch.Tensor): The input tensor for the current time step.
        - h_tt (torch.Tensor): The hidden state tensor.

        Returns:
        - t1_pred (torch.Tensor): Predicted magnitudes.
        - t1_pred_prob (torch.Tensor): Predicted probabilities.
        - h_tt (torch.Tensor): Updated hidden state.
        """

        freeze_h = self.config.get("freeze_h", "none")  # Default to "none" if key is missing

        # Compute the split index
        num_channels = h_tt.shape[1]
        split_size = num_channels // 2  # Half the channels

        if freeze_h == "hl":  # Freeze long-term memory (cell state)
            logging.debug("Freezing long-term memory (hl).")

            # Split `h_tt` into short-term (`hs_t`) and long-term (`hl_t`) components
            hs_t, hl_t_frozen = torch.split(h_tt, split_size, dim=1)

            # Run the model
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

            # Split the updated hidden state and keep the old `hl_t_frozen`
            hs_t_updated, _ = torch.split(h_tt, split_size, dim=1)

            # Concatenate the new `hs_t_updated` with the frozen `hl_t_frozen`
            h_tt = torch.cat((hs_t_updated, hl_t_frozen), dim=1)

        elif freeze_h == "hs":  # Freeze short-term memory
            logging.debug("Freezing short-term memory (hs).")

            # Split into `hs_t_frozen` and `hl_t`
            hs_t_frozen, hl_t = torch.split(h_tt, split_size, dim=1)

            # Run the model
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

            # Split the new hidden state and retain the frozen `hs_t_frozen`
            _, hl_t_updated = torch.split(h_tt, split_size, dim=1)

            # Concatenate `hs_t_frozen` with `hl_t_updated`
            h_tt = torch.cat((hs_t_frozen, hl_t_updated), dim=1)

        elif freeze_h == "all":  # Freeze both short-term and long-term memory
            logging.debug("Freezing both hs and hl.")
            t1_pred, t1_pred_class, _ = self.model(t0, h_tt)  # Do not update h_tt

        elif freeze_h == "none":  # No freezing, use normal hidden state update
            logging.debug("Not freezing any memory.")
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

        elif freeze_h == "random":  # Randomly freeze some parts
            logging.debug("Random freezing mode activated.")

            # Run model first to get new `h_tt_new`
            t1_pred, t1_pred_class, h_tt_new = self.model(t0, h_tt)

            # Split the tensors into four parts
            split_size = num_channels // 8  # Split into 8 parts
            h_tt_slices_old = torch.split(h_tt, split_size, dim=1)
            h_tt_slices_new = torch.split(h_tt_new, split_size, dim=1)

            # Randomly choose whether to keep the old or new part
            h_tt = torch.cat(
                [old if torch.rand(1) < 0.5 else new for old, new in zip(h_tt_slices_old, h_tt_slices_new)],
                dim=1
            )

        else:
            raise ValueError(f"Invalid freeze_h option: {freeze_h}. Must be one of ['hl', 'hs', 'all', 'none', 'random'].")

        return t1_pred, t1_pred_class, h_tt



    def predict(self, full_tensor: torch.Tensor, sample_idx: int, is_evaluation: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """ Predicts a sequence using the HydraNet model.

        Args:
        - full_tensor (torch.Tensor): Input tensor (batch, time, channels, H, W).
        - sample_idx (int): Current sample index for posterior sampling.
        - is_evaluation (bool): Whether running in evaluation mode.

        Returns:
        - pred_magnitudes (List[np.ndarray]): List of predicted magnitudes.
        - pred_probabilities (List[np.ndarray]): List of predicted probabilities.
        """
        logging.info(f"▶️  Starting prediction | Posterior Sample {sample_idx + 1}/{self.config['test_samples']}")

        full_tensor = full_tensor.to(self.device)  # Move tensor once to avoid redundant operations
        _, seq_len, _, H, W = full_tensor.shape  # Extract dynamic shape

        # Initialize hidden state
        h_tt = self.model.init_hTtime(hidden_channels=self.model.base, H=H, W=W).float().to(self.device)

        # Define sequence lengths based on evaluation mode
        if is_evaluation:
            full_seq_len = seq_len - 1
            in_sample_seq_len = seq_len - 1 - self.config["time_steps"]
        else:
            full_seq_len = seq_len - 1 + self.config["time_steps"]
            in_sample_seq_len = seq_len - 1

        pred_magnitudes = []
        pred_probabilities = []

        pred_magnitudes_zstack = np.zeros((self.config["time_steps"], self.config['input_channels'], H, W)) # the first dimension is the month and the second is the number of channels - should be variable
        pred_probabilities_zstack = np.zeros((self.config["time_steps"], self.config['input_channels'], H, W)) # the first dimension is the month and the second is the number of channels - should be variable

        out_of_sample_month = 0

        for t in range(full_seq_len):
            phase = "🔹 In-sample" if t < in_sample_seq_len else "🔸 Out-of-sample"
            progress = f"[{t + 1}/{full_seq_len}]"
            sys.stdout.write(f"\r{phase} | Month: {progress}  ")
            sys.stdout.flush()

            if t < in_sample_seq_len:
                t0 = full_tensor[:, t]
                t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)
            else:
                t0 = t1_pred.detach()
                t1_pred, t1_pred_class, h_tt = self.execute_freeze_h_option(t0, h_tt)
                t1_pred_class = torch.sigmoid(t1_pred_class)


                #print(f"t1_pred: {t1_pred.shape}")
                #print(f"t1_pred_class: {t1_pred_class.shape}")


                # ✅ Fix: Detach before calling .numpy()
                pred_magnitudes.append(t1_pred.cpu().detach().numpy().squeeze())
                pred_probabilities.append(t1_pred_class.cpu().detach().numpy().squeeze())


# ---------------------------------------
                # also store the predictions in a zstack

                pred_magnitudes_zstack[out_of_sample_month, :, :, :] = t1_pred.cpu().detach().numpy().squeeze()
                pred_probabilities_zstack[out_of_sample_month, :, :, :] = t1_pred_class.cpu().detach().numpy().squeeze()

                out_of_sample_month += 1

                #print(f"pred_magnitudes_zstack: {pred_magnitudes_zstack.shape}")
                #print(f"pred_probabilities_zstack: {pred_probabilities_zstack.shape}")

# ---------------------------------------

        print("\n✅ Prediction complete!")  # New line after progress updates finish

        return pred_magnitudes, pred_probabilities, pred_magnitudes_zstack, pred_probabilities_zstack


    def generate_posterior_samples(self, views_vol: torch.Tensor) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], np.ndarray, torch.Tensor]:
        """ Generates multiple posterior samples using Monte Carlo Dropout inference.

        Args:
        - views_vol (torch.Tensor): Input views data.

        Returns:
        - posterior_magnitudes (List[List[np.ndarray]]): Posterior samples of predicted magnitudes.
        - posterior_probabilities (List[List[np.ndarray]]): Posterior samples of predicted probabilities.
        - out_of_sample_vol (np.ndarray): Out-of-sample ground truth volume.
        - metadata_tensor (torch.Tensor): Metadata tensor.
        """

        logging.info(f"Drawing {self.config['test_samples']} posterior samples...")

        # Load the input tensor and metadata
        full_tensor, metadata_tensor = get_full_tensor(views_vol, self.config) # rename when refactoring

        # Move full_tensor to device once to avoid redundant memory transfers
        full_tensor = full_tensor.to(self.device)
        _, seq_len, _, H, W = full_tensor.shape  # Extract dynamic shape

        # Extract out-of-sample ground truth volume
        # out_of_sample_vol = full_tensor[:, -self.config["time_steps"]:].cpu().numpy()

        #posterior_magnitudes = []
        #posterior_probabilities = []

        posterior_magnitudes_zstack = np.zeros((self.config["time_steps"], H, W, self.config['input_channels'], self.config['test_samples']))
        posterior_probabilities_zstack = np.zeros((self.config["time_steps"], H, W, self.config['input_channels'], self.config['test_samples']))

        for sample_idx in range(self.config["test_samples"]):
            if sample_idx % 10 == 0:
                logging.info(f"Processing posterior sample {sample_idx + 1}/{self.config['test_samples']}")

            #pred_magnitudes, pred_probabilities = self.predict(full_tensor, sample_idx)
            #pred_magnitudes, pred_probabilities, pred_magnitudes_zstack, pred_probabilities_zstack = self.predict(full_tensor, sample_idx)
            _, _, pred_magnitudes_zstack, pred_probabilities_zstack = self.predict(full_tensor, sample_idx)

            #posterior_magnitudes.append(pred_magnitudes)
            #posterior_probabilities.append(pred_probabilities)

#            print("posterior_magnitudes_pred_zstack: ", pred_magnitudes_zstack.shape)
 #           print("posterior_probabilities_pred_zstack: ", pred_probabilities_zstack.shape)

 #           print("posterior_magnitudes_pred_zstack_reshaped: ", np.expand_dims(pred_magnitudes_zstack.transpose(0, 2, 3, 1), axis=-1).shape)
#            print("posterior_probabilities_pred_zstack_reshaped: ", np.expand_dims(pred_probabilities_zstack.transpose(0, 2, 3, 1), axis=-1).shape)

#            print("posterior_magnitudes_zstack: ", posterior_magnitudes_zstack.shape)
#            print("posterior_probabilities_zstack: ", posterior_probabilities_zstack.shape)

            # Store the predictions in a zstack
            posterior_magnitudes_zstack[:, :, :, :, sample_idx] = pred_magnitudes_zstack.transpose(0, 2, 3, 1)
            posterior_probabilities_zstack[:, :, :, :, sample_idx] = pred_probabilities_zstack.transpose(0, 2, 3, 1)



            # combine the two zstacks so 6 features are in the same dimension
            posterior_zstack = np.concatenate([posterior_magnitudes_zstack, posterior_probabilities_zstack], axis=-2)
            metadata_zstack = metadata_tensor.numpy()[:,-self.config['time_steps']:,:,:,:].transpose(1, 3, 4, 2, 0)
            
#            print("posterior_zstack: ", posterior_zstack.shape)

#            print("metadata_tensor: ", metadata_tensor.shape)

            #metadata_reshaped = metadata_tensor.numpy()[:,-self.config['time_steps']:,:,:,:].transpose(1, 3, 4, 2, 0)
            #metadata_repeated = np.repeat(metadata_reshaped, self.config['test_samples'], axis=-1)

            #print("metadata_repeated: ", metadata_repeated.shape)

            #zstack_combined = np.concatenate([metadata_repeated, posterior_zstack], axis=-2)

            #print("zstack_combined: ", zstack_combined.shape) 
                  

        # return posterior_magnitudes, posterior_probabilities, out_of_sample_vol, metadata_tensor, posterior_zstack, metadata_zstack
        return posterior_zstack, metadata_zstack

