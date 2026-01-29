import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

from views_hydranet.utils.utils import get_full_tensor

logger = logging.getLogger(__name__)


class HydraNetInference:
    """Handles inference with the HydraNet model.

    Includes model loading, inference execution, and posterior sampling using
    Monte Carlo Dropout for uncertainty estimation.
    """

    def __init__(
        self, model: Module, config: dict, device: Optional[str] = None
    ) -> None:
        """Initializes the inference pipeline for HydraNet.

        Args:
            model: The trained PyTorch model for inference.
            config: Configuration settings for inference.
            device: The device to run inference on ('cuda' or 'cpu').
                If not specified, it is automatically detected.

        Raises:
            TypeError: If model or config are of incorrect types.
        """
        # Step 1: Determine the best available device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")

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

        logger.info("HydraNetInference initialized successfully.")

    def _apply_dropout(self, module: torch.nn.Module) -> None:
        """Applies dropout during inference.

        This enables approximate Bayesian uncertainty estimation (MC Dropout)
        by keeping dropout layers in training mode during inference.
        """
        if isinstance(module, torch.nn.Dropout):
            module.train()


    def execute_freeze_h_option(
        self, t0: torch.Tensor, h_tt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handles the freezing of hidden state (`h_tt`) based on configuration.

        This function selectively freezes short-term (`hs`) or long-term (`hl`) memory,
        or both, based on the `config["freeze_h"]` setting.

        Args:
            t0: The input tensor for the current time step.
            h_tt: The hidden state tensor.

        Returns:
            A tuple containing:
                - t1_pred: Predicted magnitudes.
                - t1_pred_class: Predicted probabilities (pre-sigmoid if frozen).
                - h_tt: Updated (or partially frozen) hidden state.

        Raises:
            ValueError: If an invalid freeze_h option is provided.
        """

        freeze_h = self.config.get("freeze_h", "none")  # Default to "none" if key is missing

        # Compute the split index
        num_channels = h_tt.shape[1]
        split_size = num_channels // 2  # Half the channels

        if freeze_h == "hl":  # Freeze long-term memory (cell state)
            logger.debug("Freezing long-term memory (hl).")

            # Split `h_tt` into short-term (`hs_t`) and long-term (`hl_t`) components
            _, hl_t_frozen = torch.split(h_tt, split_size, dim=1)

            # Run the model
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

            # Split the updated hidden state and keep the old `hl_t_frozen`
            hs_t_updated, _ = torch.split(h_tt, split_size, dim=1)

            # Concatenate the new `hs_t_updated` with the frozen `hl_t_frozen`
            h_tt = torch.cat((hs_t_updated, hl_t_frozen), dim=1)

        elif freeze_h == "hs":  # Freeze short-term memory
            logger.debug("Freezing short-term memory (hs).")

            # Split into `hs_t_frozen` and `hl_t`
            hs_t_frozen, _ = torch.split(h_tt, split_size, dim=1)

            # Run the model
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

            # Split the new hidden state and retain the frozen `hs_t_frozen`
            _, hl_t_updated = torch.split(h_tt, split_size, dim=1)

            # Concatenate `hs_t_frozen` with `hl_t_updated`
            h_tt = torch.cat((hs_t_frozen, hl_t_updated), dim=1)

        elif freeze_h == "all":  # Freeze both short-term and long-term memory
            logger.debug("Freezing both hs and hl.")
            t1_pred, t1_pred_class, _ = self.model(t0, h_tt)  # Do not update h_tt

        elif freeze_h == "none":  # No freezing, use normal hidden state update
            logger.debug("Not freezing any memory.")
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

        elif freeze_h == "random":  # Randomly freeze some parts
            logger.debug("Random freezing mode activated.")

            # Run model first to get new `h_tt_new`
            t1_pred, t1_pred_class, h_tt_new = self.model(t0, h_tt)

            # Split the tensors into eight parts
            split_size_small = num_channels // 8
            h_tt_slices_old = torch.split(h_tt, split_size_small, dim=1)
            h_tt_slices_new = torch.split(h_tt_new, split_size_small, dim=1)

            # Randomly choose whether to keep the old or new part
            h_tt = torch.cat(
                [
                    old if torch.rand(1) < 0.5 else new
                    for old, new in zip(h_tt_slices_old, h_tt_slices_new)
                ],
                dim=1,
            )

        else:
            raise ValueError(
                f"Invalid freeze_h option: {freeze_h}. "
                "Must be one of ['hl', 'hs', 'all', 'none', 'random']."
            )

        return t1_pred, t1_pred_class, h_tt

    def predict(
        self,
        full_tensor: torch.Tensor,
        sample_idx: int,
        is_evaluation: bool = True,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts a sequence using the HydraNet model.

        Args:
            full_tensor: Input tensor (batch, time, channels, H, W).
            sample_idx: Current sample index for posterior sampling.
            is_evaluation: Whether running in evaluation mode.
            pbar: Optional progress bar to update.

        Returns:
            A tuple containing magnitudes and probabilities zstacks.
        """
        full_tensor = full_tensor.to(self.device)
        _, seq_len, _, H, W = full_tensor.shape

        # Initialize hidden state
        h_tt = (
            self.model.init_hTtime(hidden_channels=self.model.base, H=H, W=W)
            .float()
            .to(self.device)
        )

        # Define sequence lengths
        if is_evaluation:
            full_seq_len = seq_len - 1
            in_sample_seq_len = seq_len - 1 - self.config["time_steps"]
        else:
            full_seq_len = seq_len - 1 + self.config["time_steps"]
            in_sample_seq_len = seq_len - 1

        pred_magnitudes_zstack = np.zeros(
            (self.config["time_steps"], self.config["input_channels"], H, W)
        )
        pred_probabilities_zstack = np.zeros(
            (self.config["time_steps"], self.config["input_channels"], H, W)
        )

        out_of_sample_month = 0

        for t in range(full_seq_len):
            if pbar:
                pbar.set_description(f"Drawing Samples | Sample {sample_idx+1} | Step {t+1}/{full_seq_len}")
            
            if t < in_sample_seq_len:
                t0 = full_tensor[:, t]
                t1_pred, _, h_tt = self.model(t0, h_tt)
            else:
                t0 = t1_pred.detach()
                t1_pred, t1_pred_class, h_tt = self.execute_freeze_h_option(t0, h_tt)
                
                # --- PANIC CHECK: Detect explosion ---
                if not torch.isfinite(t1_pred).all():
                    logger.error(f"!!! MODEL EXPLODED at sequence step {t} !!!")
                    pred_magnitudes_zstack[out_of_sample_month:] = np.nan
                    pred_probabilities_zstack[out_of_sample_month:] = np.nan
                    break
                # --- END PANIC CHECK ---

                t1_pred_class = torch.sigmoid(t1_pred_class)

                pred_magnitudes_zstack[out_of_sample_month, :, :, :] = (
                    t1_pred.cpu().detach().numpy().squeeze()
                )
                pred_probabilities_zstack[out_of_sample_month, :, :, :] = (
                    t1_pred_class.cpu().detach().numpy().squeeze()
                )

                out_of_sample_month += 1
            
            if pbar:
                pbar.update(1)

        return pred_magnitudes_zstack, pred_probabilities_zstack


    def generate_posterior_samples(
        self, views_vol: torch.Tensor, is_evaluation: bool = True, window_info: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates multiple posterior samples using Monte Carlo Dropout.

        Args:
            views_vol: Input volume tensor.
            is_evaluation: Whether running in evaluation mode.
            window_info: Optional string to display in progress bar (e.g., "Origin 1/12").

        Returns:
            A tuple containing:
                - posterior_zstack: Concatenated magnitudes and probabilities.
                - metadata_zstack: Transposed metadata tensor.
        """
        full_tensor, metadata_tensor = get_full_tensor(views_vol, self.config)
        full_tensor = full_tensor.to(self.device)
        _, seq_len, _, H, W = full_tensor.shape

        # Define full_seq_len based on logic in predict()
        if is_evaluation:
            full_seq_len = seq_len - 1
        else:
            full_seq_len = seq_len - 1 + self.config["time_steps"]

        # Pre-allocate memory
        posterior_magnitudes_zstack = np.zeros(
            (
                self.config["time_steps"],
                H,
                W,
                self.config["input_channels"],
                self.config["test_samples"],
            ),
            dtype=np.float32,
        )
        posterior_probabilities_zstack = np.zeros_like(posterior_magnitudes_zstack)

        total_inference_steps = self.config["test_samples"] * full_seq_len
        
        desc_prefix = f"[{window_info}] " if window_info else ""

        with tqdm(
            total=total_inference_steps,
            desc=f"{desc_prefix}🎲 Drawing Posterior Samples",
            unit="step",
            leave=False, # Don't clutter the terminal, the manager has the main bar
        ) as pbar:
            for sample_idx in range(self.config["test_samples"]):
                pred_magnitudes_zstack, pred_probabilities_zstack = self.predict(
                    full_tensor, sample_idx, is_evaluation=is_evaluation, pbar=pbar
                )

                # Store slices directly without concatenation
                posterior_magnitudes_zstack[:, :, :, :, sample_idx] = (
                    pred_magnitudes_zstack.transpose(0, 2, 3, 1)
                )
                posterior_probabilities_zstack[:, :, :, :, sample_idx] = (
                    pred_probabilities_zstack.transpose(0, 2, 3, 1)
                )

        # Concatenate only once at the end
        posterior_zstack = np.concatenate(
            [posterior_magnitudes_zstack, posterior_probabilities_zstack], axis=-2
        )
        metadata_zstack = (
            metadata_tensor.numpy()[:, -self.config["time_steps"] :, :, :, :]
            .transpose(1, 3, 4, 2, 0)
        )

        return posterior_zstack, metadata_zstack