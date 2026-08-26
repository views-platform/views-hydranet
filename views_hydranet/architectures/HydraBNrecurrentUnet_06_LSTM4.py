from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from views_hydranet.architectures.locked_dropout import LockedDropout
from views_hydranet.distributions import resolve_family
from views_hydranet.utils.quantile_head import init_quantile_conv_, monotone_quantiles


def _family_activation(family):
    """Wrap a ``DistributionFamily.activate`` for the channel-first reg head (ADR-067).

    The head emits ``[B, n_params, H, W]`` per target, but ``activate`` takes the params in the
    LAST dim (``[..., n_params]``) — permute in, activate, permute back (the quantile-head idiom).
    """

    def _act(x):
        p = family.activate(x.permute(0, 2, 3, 1))
        return p.permute(0, 3, 1, 2).contiguous()

    return _act


class ModelOutput(NamedTuple):
    """
    Typed return value for HydraBNUNet06_LSTM4.forward().

    Fields reg, cls, h_next are the primary outputs. reg_latent carries the
    pre-ReLU regression activations needed by censored losses (TobitLoss).
    Consumers that do not need latent values can ignore it — default is None.

    NOTE: Adding reg_latent as a 4th field means tuple unpacking with 3
    variables (``r, c, h = model(x, h)``) no longer works. Use named access:
    ``output.reg, output.cls, output.h_next``.
    """

    reg: torch.Tensor  # [B, n_reg, H, W] post-ReLU regression head outputs
    cls: torch.Tensor  # [B, n_cls, H, W] classification head logits
    h_next: torch.Tensor  # [B, total_hidden_channels, H, W] LSTM hidden state
    reg_latent: torch.Tensor | None = None  # [B, n_reg, H, W] pre-ReLU latent mu


class HydraBNUNet06_LSTM4(nn.Module):
    """
    Recurrent U-Net with Batch Normalization and Quad-LSTM temporal memory.

    This architecture combines spatial multi-scale processing (U-Net) with
    temporal memory (ConvLSTM). It is designed for multi-task forecasting
    (State-Based, Non-State, One-Sided) by producing multiple regression
    and classification heads.

    Architecture Highlights:
        - 4 Parallel ConvLSTM cells managing short and long-term memory.
        - 2 Encoder levels with Batch Normalization and Max Pooling.
        - 6 Independent decoder heads (3 Regression, 3 Classification).
        - Skip-connections at both the input level and U-Net levels.

    Attributes:
        base (int): The number of hidden channels used as the architectural baseline.
    """

    def __init__(
        self,
        input_channels,
        total_hidden_channels,
        output_channels,
        dropout_rate,
        output_distribution="standard",
        n_static_channels=0,
        static_top_skip=True,
        reg_activation=None,
        n_quantiles=None,
        state_channels=None,
    ):
        """
        Initializes the HydraNet architecture.

        Args:
            input_channels (int): Number of input feature channels.
            total_hidden_channels (int): Capacity of the recurrent memory.
                                         Must be divisible by 8.
            output_channels (int): Number of channels per head (usually 1).
            dropout_rate (float): Probability of dropout for regularization.
            output_distribution (str): regression-head output distribution/activation —
                "standard" (ReLU, default, pre-#100 behavior) or a family/head name
                (nb, zinb, quantile, hurdle_nb, hurdle_lognormal, hurdle_shrinkage), which
                selects the ADR-067 family activation / head param count.
        """
        super().__init__()

        # #100: hurdle-NB needs a count-space mean mu at the output; softplus keeps it
        # sub-exponential. Default "standard" => ReLU => byte-identical to pre-#100.
        self.output_distribution = output_distribution
        # ADR-067 strangler-fig: if output_distribution names a registered distribution family
        # (nb/zinb), the head sizes reg channels to family.n_params and the FAMILY owns activation.
        # resolve_family returns None for every legacy value, so those fall through to the
        # untouched legacy path below (byte-identical). New families add zero new branches here.
        self._family = resolve_family(output_distribution)
        # Default keys off output_distribution: ALL hurdle bodies => softplus, "standard" => relu.
        # reg_activation overrides it (Exp B: decouple emit activation from the loss/likelihood).
        # C-178: a ReLU output head dies on rare targets under the hurdle mask (pre-activation
        # drifts 100% negative => ReLU==0 with zero gradient => unrecoverable). softplus is always
        # positive with non-zero gradient, so IT cannot die. hurdle_nb already used softplus;
        # extend to all hurdle bodies. "standard" keeps relu (byte-identical to pre-#100).
        #
        # C-313: that last clause is about the ACTIVATION, not the path. For the family heads,
        # `nb_core._clamp` applies `clamp_min(1e-6)` downstream, and `clamp_min` passes exactly
        # zero gradient below its floor — so the path CAN die, at raw <= log(expm1(1e-6)) ~ -13.82.
        # Latent on the current vehicle (the trained incumbent's min raw_mu is -3.81, and the
        # gradient there points away from the edge), pinned by
        # tests/distributions/test_gradient_dead_zones.py. Do not read this comment as a guarantee
        # that the emitted mu/theta cannot reach a zero-gradient state.
        if self._family is not None:
            self._reg_activation = _family_activation(self._family)
        elif reg_activation == "softplus":
            self._reg_activation = F.softplus
        elif reg_activation == "relu":
            self._reg_activation = F.relu
        else:
            self._reg_activation = (
                F.softplus if output_distribution.startswith("hurdle") else F.relu
            )

        # Quantile head (2026-07 build): each reg head emits K monotone quantiles
        # (cumulative-softplus
        # in log1p space) instead of a single activated mu. output_channels stays the
        # feedback/class
        # width (1); n_quantiles (K) sizes ONLY the reg heads, so the AR-feedback invariant
        # (input_channels == 3*output_channels + static) is untouched. See quantile_head.py.
        self._is_quantile = output_distribution == "quantile"
        self.n_quantiles = n_quantiles
        if self._is_quantile:
            if not n_quantiles or n_quantiles < 2:
                raise ValueError("output_distribution='quantile' requires n_quantiles >= 2")

            # [B, K, H, W] -> monotone along the channel (quantile) axis
            def _monotone_channels(x):
                q = monotone_quantiles(x.permute(0, 2, 3, 1))  # [B, H, W, K]
                return q.permute(0, 3, 1, 2).contiguous()

            self._reg_activation = _monotone_channels
        # A family emits n_params reg channels per target (nb=2 [mu,theta], zinb=3 [+pi]); the
        # quantile head emits K; every other legacy head emits output_channels (the AR-feedback
        # width, unchanged so the invariant input_channels == 3*output_channels + static holds).
        if self._family is not None:
            reg_out_ch = self._family.n_params
        else:
            reg_out_ch = n_quantiles if self._is_quantile else output_channels

        # ADR-061 top-skip: the last n_static_channels of the input are static (e.g. coordinates);
        # re-injected raw at each decoder head's full-resolution dec_conv1. 0 => byte-identical.
        # `static_top_skip=False` keeps statics in the ENCODER input but drops them from the
        # top-skip re-injection (C-228: raw statics at the gate head's dec_conv1 collapse AP). The
        # encoder still receives them via input_channels; only the head re-injection is removed.
        self.n_static = n_static_channels
        #: dynamic (non-static) input channels — the conflict field. Subclass seams slice these as
        #: `x[:, :self.n_dynamic]`; statics are the LAST n_static, which is why the two differ.
        self.n_dynamic = input_channels - n_static_channels
        self.static_top_skip = static_top_skip
        topskip_static = n_static_channels if static_top_skip else 0

        kernel_size = 3
        # byte-identical to the pre-seam model (pinned by test_architecture_registry.py);
        # the WideMemory candidate passes a wider state to vary the memory ALONE.
        # the same number, which is why raising `total_hidden_channels` widens the whole network
        # rather than just its memory. `state_channels=None` keeps them identical and therefore
        # byte-identical to the pre-seam model (pinned by test_architecture_registry.py); the
        # WideMemory candidate passes a wider state to vary the memory ALONE.
        base = total_hidden_channels
        state = total_hidden_channels if state_channels is None else int(state_channels)
        if state % 8:
            raise ValueError(
                f"state_channels={state} is not divisible by 8 (4 short-term + 4 long-term "
                "groups); blend_recurrent_state would silently mis-assign memory types."
            )
        lstm_padding = kernel_size // 2

        num_lstm_cells = 4
        num_lstm_state_layers = int(state / (num_lstm_cells * 2))

        #: The RECURRENT width. Callers size the state from this (`init_hTtime(model.base, ...)`),
        #: so it must be the state width, not the conv width.
        self.base = state
        self.conv_base = base

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(
            input_channels + int(state / 2),
            base,
            kernel_size,
            padding=1,
            bias=False,
        )

        self.bn_enc_conv0 = nn.BatchNorm2d(base)
        self.pool0 = nn.MaxPool2d(2, 2, padding=0)

        self.enc_conv1 = nn.Conv2d(base, base * 2, kernel_size, padding=1, bias=False)
        self.bn_enc_conv1 = nn.BatchNorm2d(base * 2)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(base * 2, base * 4, kernel_size, padding=1, bias=False)
        self.bn_bottleneck_conv = nn.BatchNorm2d(base * 4)

        # HEAD1 reg
        self.upsample0_head1_reg = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv0_head1_reg = nn.Conv2d(
            base * 4, base * 2, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv0_head1_reg = nn.BatchNorm2d(base * 2)

        self.upsample1_head1_reg = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv1_head1_reg = nn.Conv2d(
            base * 2 + topskip_static, base, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv1_head1_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head1_reg = nn.Conv2d(base, reg_out_ch, kernel_size, padding=1)

        # HEAD1 class
        self.upsample0_head1_class = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv0_head1_class = nn.Conv2d(
            base * 4, base * 2, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv0_head1_class = nn.BatchNorm2d(base * 2)

        self.upsample1_head1_class = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv1_head1_class = nn.Conv2d(
            base * 2 + topskip_static, base, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv1_head1_class = nn.BatchNorm2d(base)

        self.dec_conv4_head1_class = nn.Conv2d(base, output_channels, 3, padding=1)

        # HEAD2 reg
        self.upsample0_head2_reg = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv0_head2_reg = nn.Conv2d(
            base * 4, base * 2, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv0_head2_reg = nn.BatchNorm2d(base * 2)

        self.upsample1_head2_reg = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv1_head2_reg = nn.Conv2d(
            base * 2 + topskip_static, base, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv1_head2_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head2_reg = nn.Conv2d(base, reg_out_ch, 3, padding=1)

        # HEAD2 class
        self.upsample0_head2_class = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv0_head2_class = nn.Conv2d(
            base * 4, base * 2, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv0_head2_class = nn.BatchNorm2d(base * 2)

        self.upsample1_head2_class = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv1_head2_class = nn.Conv2d(
            base * 2 + topskip_static, base, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv1_head2_class = nn.BatchNorm2d(base)

        self.dec_conv4_head2_class = nn.Conv2d(base, output_channels, kernel_size, padding=1)

        # HEAD3 reg
        self.upsample0_head3_reg = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv0_head3_reg = nn.Conv2d(
            base * 4, base * 2, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv0_head3_reg = nn.BatchNorm2d(base * 2)

        self.upsample1_head3_reg = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv1_head3_reg = nn.Conv2d(
            base * 2 + topskip_static, base, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv1_head3_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head3_reg = nn.Conv2d(base, reg_out_ch, kernel_size, padding=1)

        # Informed init (C-199 / C-203): seed each reg head's bias from the family's own recipe so
        # theta/pi start away from the saturated dead-zone (family DEFAULT priors — the head has no
        # data; data-derived priors are a later refinement). Narrow-fan init for the quantile heads
        # (channel 0 = base, 1: = pre-softplus gaps) so the cumulated softplus opens narrow.
        _reg_convs = (self.dec_conv4_head1_reg, self.dec_conv4_head2_reg, self.dec_conv4_head3_reg)
        if self._family is not None:
            _bias = self._family.initial_raw_bias()
            for _c in _reg_convs:
                with torch.no_grad():
                    _c.bias.copy_(_bias.to(_c.bias.dtype))
        elif self._is_quantile:
            for _c in _reg_convs:
                init_quantile_conv_(_c)

        # HEAD3 class
        self.upsample0_head3_class = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv0_head3_class = nn.Conv2d(base * 4, base * 2, 3, padding=1, bias=False)
        self.bn_dec_conv0_head3_class = nn.BatchNorm2d(base * 2)

        self.upsample1_head3_class = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv1_head3_class = nn.Conv2d(
            base * 2 + topskip_static, base, kernel_size, padding=1, bias=False
        )
        self.bn_dec_conv1_head3_class = nn.BatchNorm2d(base)

        self.dec_conv4_head3_class = nn.Conv2d(base, output_channels, kernel_size, padding=1)

        # Dropout — per-site LockedDropout (ADR-057; C-128 fix). Each of the 15
        # dropout sites gets its OWN instance, so locked (MC-dropout inference)
        # masks are independent PER LAYER. The previous single shared instance
        # cached masks by (shape,device,dtype), so same-shaped sites collided on
        # one mask — correlated epistemic dropout, not the per-layer masks
        # Gal & Ghahramani intend. set/reset_locked_dropout iterate self.modules(),
        # so every site is handled; LockedDropout has no params/buffers, so
        # existing artifacts load unchanged. Training is unchanged (locked=False
        # → a fresh mask every call, byte-identical to nn.Dropout, same RNG order).
        _dropout_sites = (
            "e0s",
            "e1s",
            "b",
            "h1_reg_d0",
            "h1_reg",
            "h1_class_d0",
            "h1_class",
            "h2_reg_d0",
            "h2_reg",
            "h2_class_d0",
            "h2_class",
            "h3_reg_d0",
            "h3_reg",
            "h3_class_d0",
            "h3_class",
        )
        self.dropout = nn.ModuleDict(
            {site: LockedDropout(p=dropout_rate) for site in _dropout_sites}
        )

        # LSTM parameters initialization (gate weights for the 4 stacked ConvLSTM cells).
        self.Wxi_1 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whi_1 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxf_1 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whf_1 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxc_1 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whc_1 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxo_1 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Who_1 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )

        self.Wxi_2 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whi_2 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxf_2 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whf_2 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxc_2 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whc_2 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxo_2 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Who_2 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )

        self.Wxi_3 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whi_3 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxf_3 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whf_3 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxc_3 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whc_3 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxo_3 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Who_3 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )

        self.Wxi_4 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whi_4 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxf_4 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whf_4 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxc_4 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Whc_4 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )
        self.Wxo_4 = nn.Conv2d(
            input_channels, num_lstm_state_layers, kernel_size, padding=lstm_padding, bias=True
        )
        self.Who_4 = nn.Conv2d(
            num_lstm_state_layers,
            num_lstm_state_layers,
            kernel_size,
            padding=lstm_padding,
            bias=True,
        )

    def _topskip(self, e0s, coords, x):
        """Build the full-resolution skip fed to every head's ``dec_conv1``.

        **An extension seam, not a behaviour change.** The default is exactly the ADR-061 rule it
        replaced — concat the raw statics when there are any, otherwise pass ``e0s`` through — and
        `tests/architectures/test_architecture_registry.py` pins that the incumbent stays
        byte-identical through it.

        It exists because three bake-off candidates (DynamicTopSkip, FiLMSkip, DualStream) differ
        from the incumbent ONLY here, and the alternative was four near-identical copies of a
        90-line `forward` with six spelled-out decoder paths. Subclasses override this one method.

        Args:
            e0s: the full-resolution encoder feature, ``[B, base, H, W]``.
            coords: the raw static channels, or ``None`` when there are none.
            x: the encoder input, ``[B, in + 4*state, H, W]`` — the raw dynamic channels are its
                FIRST ``self.n_dynamic``, which is what the dynamic-skip candidates consume.
        """
        return torch.cat([e0s, coords], 1) if coords is not None else e0s

    def forward(self, x, h):
        """
        Performs a forward pass for a single time step.

        Args:
            x (torch.Tensor): Input tensor [batch, channels, H, W].
            h (torch.Tensor): Combined hidden state [batch, total_hidden_channels, H, W].

        Returns:
            ModelOutput: NamedTuple with `reg`, `cls`, `h_next` fields. Supports
                tuple unpacking for backward compatibility.
        """

        # ADR-061 top-skip: capture the raw static channels (the last n_static of the input)
        # at full resolution, before x is concatenated with the hidden state. None => no top-skip.
        coords = x[:, -self.n_static :, :, :] if (self.n_static and self.static_top_skip) else None

        # Splitting hidden state into 4 short-term and 4 long-term memory tensors.
        split_h = int(h.shape[1] / 8)
        hs_1, hs_2, hs_3, hs_4, hl_1, hl_2, hl_3, hl_4 = torch.split(h, split_h, dim=1)

        # ----------------- LSTM 1 -----------------
        i_t_1 = torch.sigmoid(self.Wxi_1(x) + self.Whi_1(hs_1))
        f_t_1 = torch.sigmoid(self.Wxf_1(x) + self.Whf_1(hs_1))
        hl_1_tilde = torch.tanh(self.Wxc_1(x) + self.Whc_1(hs_1))
        hl_1 = f_t_1 * hl_1 + i_t_1 * hl_1_tilde
        o_t_1 = torch.sigmoid(self.Wxo_1(x) + self.Who_1(hs_1))
        hs_1 = o_t_1 * torch.tanh(hl_1)

        # ----------------- LSTM 2 -----------------
        i_t_2 = torch.sigmoid(self.Wxi_2(x) + self.Whi_2(hs_2))
        f_t_2 = torch.sigmoid(self.Wxf_2(x) + self.Whf_2(hs_2))
        hl_2_tilde = torch.tanh(self.Wxc_2(x) + self.Whc_2(hs_2))
        hl_2 = f_t_2 * hl_2 + i_t_2 * hl_2_tilde
        o_t_2 = torch.sigmoid(self.Wxo_2(x) + self.Who_2(hs_2))
        hs_2 = o_t_2 * torch.tanh(hl_2)

        # ----------------- LSTM 3 -----------------
        i_t_3 = torch.sigmoid(self.Wxi_3(x) + self.Whi_3(hs_3))
        f_t_3 = torch.sigmoid(self.Wxf_3(x) + self.Whf_3(hs_3))
        hl_3_tilde = torch.tanh(self.Wxc_3(x) + self.Whc_3(hs_3))
        hl_3 = f_t_3 * hl_3 + i_t_3 * hl_3_tilde
        o_t_3 = torch.sigmoid(self.Wxo_3(x) + self.Who_3(hs_3))
        hs_3 = o_t_3 * torch.tanh(hl_3)

        # ----------------- LSTM 4 -----------------
        i_t_4 = torch.sigmoid(self.Wxi_4(x) + self.Whi_4(hs_4))
        f_t_4 = torch.sigmoid(self.Wxf_4(x) + self.Whf_4(hs_4))
        hl_4_tilde = torch.tanh(self.Wxc_4(x) + self.Whc_4(hs_4))
        hl_4 = f_t_4 * hl_4 + i_t_4 * hl_4_tilde
        o_t_4 = torch.sigmoid(self.Wxo_4(x) + self.Who_4(hs_4))
        hs_4 = o_t_4 * torch.tanh(hl_4)

        h = torch.cat([hs_1, hs_2, hs_3, hs_4, hl_1, hl_2, hl_3, hl_4], 1)
        x = torch.cat([x, hs_1, hs_2, hs_3, hs_4], 1)

        # encoder
        e0s_ = F.relu(self.bn_enc_conv0(self.enc_conv0(x)))
        e0s = self.dropout["e0s"](e0s_)
        # ADR-061: append the raw coords to the full-resolution skip fed to every head's decision
        # layer (dec_conv1). Encoder downsampling below uses e0s (no coords). None => unchanged.
        e0s_topskip = self._topskip(e0s, coords, x)
        e0 = self.pool0(e0s)
        e1s = self.dropout["e1s"](F.relu(self.bn_enc_conv1(self.enc_conv1(e0))))
        e1 = self.pool1(e1s)

        # bottleneck
        b = F.relu(self.bn_bottleneck_conv(self.bottleneck_conv(e1)))
        b = self.dropout["b"](b)

        # DECODERS (H1, H2, H3)
        # H1 reg
        H1_d0 = F.relu(
            self.bn_dec_conv0_head1_reg(
                self.dec_conv0_head1_reg(torch.cat([self.upsample0_head1_reg(b), e1s], 1))
            )
        )
        H1_d0 = self.dropout["h1_reg_d0"](H1_d0)
        H1_d1 = F.relu(
            self.bn_dec_conv1_head1_reg(
                self.dec_conv1_head1_reg(
                    torch.cat([self.upsample1_head1_reg(H1_d0), e0s_topskip], 1)
                )
            )
        )
        H1_reg = self.dropout["h1_reg"](H1_d1)
        H1_reg = self.dec_conv4_head1_reg(H1_reg)
        out_reg1 = self._reg_activation(H1_reg)

        # H1 class
        H1_d0 = F.relu(
            self.bn_dec_conv0_head1_class(
                self.dec_conv0_head1_class(torch.cat([self.upsample0_head1_class(b), e1s], 1))
            )
        )
        H1_d0 = self.dropout["h1_class_d0"](H1_d0)
        H1_d1 = F.relu(
            self.bn_dec_conv1_head1_class(
                self.dec_conv1_head1_class(
                    torch.cat([self.upsample1_head1_class(H1_d0), e0s_topskip], 1)
                )
            )
        )
        H1_class = self.dropout["h1_class"](H1_d1)
        H1_class = self.dec_conv4_head1_class(H1_class)
        out_class1 = H1_class

        # H2 reg
        H2_d0 = F.relu(
            self.bn_dec_conv0_head2_reg(
                self.dec_conv0_head2_reg(torch.cat([self.upsample0_head2_reg(b), e1s], 1))
            )
        )
        H2_d0 = self.dropout["h2_reg_d0"](H2_d0)
        H2_d1 = F.relu(
            self.bn_dec_conv1_head2_reg(
                self.dec_conv1_head2_reg(
                    torch.cat([self.upsample1_head2_reg(H2_d0), e0s_topskip], 1)
                )
            )
        )
        H2_reg = self.dropout["h2_reg"](H2_d1)
        H2_reg = self.dec_conv4_head2_reg(H2_reg)
        out_reg2 = self._reg_activation(H2_reg)

        # H2 class
        H2_d0 = F.relu(
            self.bn_dec_conv0_head2_class(
                self.dec_conv0_head2_class(torch.cat([self.upsample0_head2_class(b), e1s], 1))
            )
        )
        H2_d0 = self.dropout["h2_class_d0"](H2_d0)
        H2_d1 = F.relu(
            self.bn_dec_conv1_head2_class(
                self.dec_conv1_head2_class(
                    torch.cat([self.upsample1_head2_class(H2_d0), e0s_topskip], 1)
                )
            )
        )
        H2_class = self.dropout["h2_class"](H2_d1)
        H2_class = self.dec_conv4_head2_class(H2_class)
        out_class2 = H2_class

        # H3 reg
        H3_d0 = F.relu(
            self.bn_dec_conv0_head3_reg(
                self.dec_conv0_head3_reg(torch.cat([self.upsample0_head3_reg(b), e1s], 1))
            )
        )
        H3_d0 = self.dropout["h3_reg_d0"](H3_d0)
        H3_d1 = F.relu(
            self.bn_dec_conv1_head3_reg(
                self.dec_conv1_head3_reg(
                    torch.cat([self.upsample1_head3_reg(H3_d0), e0s_topskip], 1)
                )
            )
        )
        H3_reg = self.dropout["h3_reg"](H3_d1)
        H3_reg = self.dec_conv4_head3_reg(H3_reg)
        out_reg3 = self._reg_activation(H3_reg)

        # H3 class
        H3_d0 = F.relu(
            self.bn_dec_conv0_head3_class(
                self.dec_conv0_head3_class(torch.cat([self.upsample0_head3_class(b), e1s], 1))
            )
        )
        H3_d0 = self.dropout["h3_class_d0"](H3_d0)
        H3_d1 = F.relu(
            self.bn_dec_conv1_head3_class(
                self.dec_conv1_head3_class(
                    torch.cat([self.upsample1_head3_class(H3_d0), e0s_topskip], 1)
                )
            )
        )
        H3_class = self.dropout["h3_class"](H3_d1)
        H3_class = self.dec_conv4_head3_class(H3_class)
        out_class3 = H3_class

        out_reg_latent = torch.concat([H1_reg, H2_reg, H3_reg], dim=1) if self.training else None
        out_reg = torch.concat([out_reg1, out_reg2, out_reg3], dim=1)
        out_class = torch.concat([out_class1, out_class2, out_class3], dim=1)

        return ModelOutput(reg=out_reg, cls=out_class, h_next=h, reg_latent=out_reg_latent)

    def set_locked_dropout(self, enabled: bool) -> None:
        """Enable/disable consistent-mask (variational) dropout (ADR-057).

        When enabled, each ``LockedDropout`` is put in train mode (so MC-dropout
        sampling is active at inference) and its mask is locked across forwards.
        Only the dropout submodules are switched to train mode — the rest of the
        model stays in its current (eval) mode, so BatchNorm running stats and
        the train-only ``reg_latent`` path are unaffected.
        """
        for module in self.modules():
            if isinstance(module, LockedDropout):
                module.locked = enabled
                if enabled:
                    module.train()
                    module.reset()

    def reset_locked_dropout(self) -> None:
        """Refresh all locked dropout masks — call once per posterior sample so
        each sample's autoregressive trajectory draws a fresh, internally
        consistent mask."""
        for module in self.modules():
            if isinstance(module, LockedDropout):
                module.reset()

    def init_hTtime(self, hidden_channels, H, W):
        """
        Initializes the recurrent hidden state.

        Args:
            hidden_channels (int): Must match total_hidden_channels in __init__.
            H (int): Grid height.
            W (int): Grid width.

        Returns:
            torch.Tensor: Zero-initialized state [1, hidden_channels, H, W] in float32.
        """
        hs = torch.zeros((1, hidden_channels, H, W), dtype=torch.float32)
        return hs
