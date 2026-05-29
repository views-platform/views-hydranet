from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# give everything better names at some point
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

    def __init__(self, input_channels, total_hidden_channels, output_channels, dropout_rate):
        """
        Initializes the HydraNet architecture.

        Args:
            input_channels (int): Number of input feature channels.
            total_hidden_channels (int): Capacity of the recurrent memory.
                                         Must be divisible by 8.
            output_channels (int): Number of channels per head (usually 1).
            dropout_rate (float): Probability of dropout for regularization.
        """
        super().__init__()

        kernel_size = 3
        base = total_hidden_channels
        lstm_padding = kernel_size // 2

        num_lstm_cells = 4
        num_lstm_state_layers = int(total_hidden_channels / (num_lstm_cells * 2))

        self.base = base

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(
            input_channels + int(total_hidden_channels / 2),
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
        self.dec_conv1_head1_reg = nn.Conv2d(base * 2, base, kernel_size, padding=1, bias=False)
        self.bn_dec_conv1_head1_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head1_reg = nn.Conv2d(base, output_channels, kernel_size, padding=1)

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
        self.dec_conv1_head1_class = nn.Conv2d(base * 2, base, kernel_size, padding=1, bias=False)
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
        self.dec_conv1_head2_reg = nn.Conv2d(base * 2, base, kernel_size, padding=1, bias=False)
        self.bn_dec_conv1_head2_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head2_reg = nn.Conv2d(base, output_channels, 3, padding=1)

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
        self.dec_conv1_head2_class = nn.Conv2d(base * 2, base, kernel_size, padding=1, bias=False)
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
        self.dec_conv1_head3_reg = nn.Conv2d(base * 2, base, kernel_size, padding=1, bias=False)
        self.bn_dec_conv1_head3_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head3_reg = nn.Conv2d(base, output_channels, kernel_size, padding=1)

        # HEAD3 class
        self.upsample0_head3_class = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv0_head3_class = nn.Conv2d(base * 4, base * 2, 3, padding=1, bias=False)
        self.bn_dec_conv0_head3_class = nn.BatchNorm2d(base * 2)

        self.upsample1_head3_class = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2, padding=0, output_padding=0
        )
        self.dec_conv1_head3_class = nn.Conv2d(base * 2, base, kernel_size, padding=1, bias=False)
        self.bn_dec_conv1_head3_class = nn.BatchNorm2d(base)

        self.dec_conv4_head3_class = nn.Conv2d(base, output_channels, kernel_size, padding=1)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # LSTM parameters initialization...
        # [Implementation details omitted for brevity, logic remains identical]
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

        # Splitting hidden state into 4 short-term and 4 long-term memory tensors.
        split_h = int(h.shape[1] / 8)
        hs_1, hs_2, hs_3, hs_4, hl_1, hl_2, hl_3, hl_4 = torch.split(h, split_h, dim=1)

        # ... [LSTM Logic remains identical] ...
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
        e0s = self.dropout(e0s_)
        e0 = self.pool0(e0s)
        e1s = self.dropout(F.relu(self.bn_enc_conv1(self.enc_conv1(e0))))
        e1 = self.pool1(e1s)

        # bottleneck
        b = F.relu(self.bn_bottleneck_conv(self.bottleneck_conv(e1)))
        b = self.dropout(b)

        # DECODERS (H1, H2, H3 logic remains identical)
        # H1 reg
        H1_d0 = F.relu(
            self.bn_dec_conv0_head1_reg(
                self.dec_conv0_head1_reg(torch.cat([self.upsample0_head1_reg(b), e1s], 1))
            )
        )
        H1_d0 = self.dropout(H1_d0)
        H1_d1 = F.relu(
            self.bn_dec_conv1_head1_reg(
                self.dec_conv1_head1_reg(torch.cat([self.upsample1_head1_reg(H1_d0), e0s], 1))
            )
        )
        H1_reg = self.dropout(H1_d1)
        H1_reg = self.dec_conv4_head1_reg(H1_reg)
        out_reg1 = F.relu(H1_reg)

        # H1 class
        H1_d0 = F.relu(
            self.bn_dec_conv0_head1_class(
                self.dec_conv0_head1_class(torch.cat([self.upsample0_head1_class(b), e1s], 1))
            )
        )
        H1_d0 = self.dropout(H1_d0)
        H1_d1 = F.relu(
            self.bn_dec_conv1_head1_class(
                self.dec_conv1_head1_class(torch.cat([self.upsample1_head1_class(H1_d0), e0s], 1))
            )
        )
        H1_class = self.dropout(H1_d1)
        H1_class = self.dec_conv4_head1_class(H1_class)
        out_class1 = H1_class

        # H2 reg
        H2_d0 = F.relu(
            self.bn_dec_conv0_head2_reg(
                self.dec_conv0_head2_reg(torch.cat([self.upsample0_head2_reg(b), e1s], 1))
            )
        )
        H2_d0 = self.dropout(H2_d0)
        H2_d1 = F.relu(
            self.bn_dec_conv1_head2_reg(
                self.dec_conv1_head2_reg(torch.cat([self.upsample1_head2_reg(H2_d0), e0s], 1))
            )
        )
        H2_reg = self.dropout(H2_d1)
        H2_reg = self.dec_conv4_head2_reg(H2_reg)
        out_reg2 = F.relu(H2_reg)

        # H2 class
        H2_d0 = F.relu(
            self.bn_dec_conv0_head2_class(
                self.dec_conv0_head2_class(torch.cat([self.upsample0_head2_class(b), e1s], 1))
            )
        )
        H2_d0 = self.dropout(H2_d0)
        H2_d1 = F.relu(
            self.bn_dec_conv1_head2_class(
                self.dec_conv1_head2_class(torch.cat([self.upsample1_head2_class(H2_d0), e0s], 1))
            )
        )
        H2_class = self.dropout(H2_d1)
        H2_class = self.dec_conv4_head2_class(H2_class)
        out_class2 = H2_class

        # H3 reg
        H3_d0 = F.relu(
            self.bn_dec_conv0_head3_reg(
                self.dec_conv0_head3_reg(torch.cat([self.upsample0_head3_reg(b), e1s], 1))
            )
        )
        H3_d0 = self.dropout(H3_d0)
        H3_d1 = F.relu(
            self.bn_dec_conv1_head3_reg(
                self.dec_conv1_head3_reg(torch.cat([self.upsample1_head3_reg(H3_d0), e0s], 1))
            )
        )
        H3_reg = self.dropout(H3_d1)
        H3_reg = self.dec_conv4_head3_reg(H3_reg)
        out_reg3 = F.relu(H3_reg)

        # H3 class
        H3_d0 = F.relu(
            self.bn_dec_conv0_head3_class(
                self.dec_conv0_head3_class(torch.cat([self.upsample0_head3_class(b), e1s], 1))
            )
        )
        H3_d0 = self.dropout(H3_d0)
        H3_d1 = F.relu(
            self.bn_dec_conv1_head3_class(
                self.dec_conv1_head3_class(torch.cat([self.upsample1_head3_class(H3_d0), e0s], 1))
            )
        )
        H3_class = self.dropout(H3_d1)
        H3_class = self.dec_conv4_head3_class(H3_class)
        out_class3 = H3_class

        out_reg_latent = torch.concat([H1_reg, H2_reg, H3_reg], dim=1)
        out_reg = torch.concat([out_reg1, out_reg2, out_reg3], dim=1)
        out_class = torch.concat([out_class1, out_class2, out_class3], dim=1)

        return ModelOutput(reg=out_reg, cls=out_class, h_next=h, reg_latent=out_reg_latent)

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
