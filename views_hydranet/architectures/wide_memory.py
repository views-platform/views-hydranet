"""WideMemory — widen the recurrent state without widening anything else (bake-off candidate 6).

Measured on the incumbent: **905,289 parameters, of which the ConvLSTM holds 4,160 — 0.5%** — while
the six decoder paths hold 89%. The model is overwhelmingly a per-frame decoder with a very small
recurrent memory, and that memory is the only thing carrying placement across 36 autoregressive
steps. Since *freezing the cell state* is the one rollout intervention that ever helped (M38/M39,
+0.039 AP@h18), "the memory is too small to carry anything" is a live hypothesis.

**Why this cannot be a config change.** In the incumbent `base = total_hidden_channels` sets the
recurrent width AND every conv width, so raising it in the config would widen the whole network and
confound the memory with ~900k parameters of decoder. The base class now accepts `state_channels`
(defaulting to `total_hidden_channels`, hence byte-identical when unused); this class supplies a
wider one and leaves the conv stack at the incumbent's width.
"""

from __future__ import annotations

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

#: How much wider the recurrent state is than the incumbent's: 32 -> 128 channels, i.e. 16 per LSTM
#: cell instead of 4. Hard-coded, not configurable, so an arm differs from its control in exactly
#: ONE config key (`model`) — the single-variable invariant this program runs on.
MEMORY_WIDTH_FACTOR = 4


class WideMemory(HydraBNUNet06_LSTM4):
    """The incumbent with a `MEMORY_WIDTH_FACTOR`x wider ConvLSTM state; conv stack unchanged."""

    def __init__(
        self, input_channels, total_hidden_channels, output_channels, dropout_rate, **kwargs
    ):
        kwargs.pop("state_channels", None)  # this class owns the state width
        super().__init__(
            input_channels,
            total_hidden_channels,
            output_channels,
            dropout_rate,
            state_channels=total_hidden_channels * MEMORY_WIDTH_FACTOR,
            **kwargs,
        )
