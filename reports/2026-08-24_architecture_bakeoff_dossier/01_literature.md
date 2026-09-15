# 01 — Literature (2026-08-24)

Claims below are **extracted into the library** (`papers/_claims/<id>.json`) and quoted from there.
Retrieval caveat recorded once: the library's `chunks` table is **0**, so semantic search sees titles,
abstracts and extracted claims only — never body text. Everything here comes from extracted claims.

## Load-bearing for this program

### `Zhang2019_AntiAliasedCNN` → candidate (1) AntiAliasedPool

> "Modern CNN downsampling operations (max-pooling, strided-convolution, average-pooling) **violate the
> sampling theorem and break shift-equivariance**. Max-pooling is decomposable into dense max evaluation
> + naive subsampling; inserting a low-pass anti-aliasing filter between them (MaxBlurPool) preserves
> shift-equivariance while retaining the benefits of max-pooling."

> "Shift-equivariance is **progressively lost at each downsampling layer**… all layers before the first
> downsampling are shift-equivariant; each subsequent subsampling introduces periodic-N
> shift-equivariance with doubling factor N."

> Bin-5 on ResNet50: consistency **+2.1%**, and "surprisingly improves classification accuracy by
> **+0.7–0.9% without adding learnable parameters**, serving as effective regularization."

**What we take:** our network has **two** `MaxPool2d(2,2)`, so it is shift-equivariant only up to
`enc_conv0` and loses equivariance twice thereafter. On a task where *where* is the entire gap, this is
a directly implicated mechanism, and the remedy adds **no parameters** — the cleanest possible arm.

### `Sun2019_HRNet` → candidate (5) DualStream

> "HRNet **maintains high-resolution representations throughout** the network by connecting parallel
> multi-resolution convolutions… with repeated cross-resolution fusions, **rather than recovering
> high-resolution via encoder-decoder**."

> HRNetV2-W48: 81.1% mIoU Cityscapes val vs DeepLabv3 78.5%, DeepLabv3+ 79.6%, PSPNet 79.7% — "with
> **lower computation complexity** (747.3 GFLOPs vs 1778.7 / 2017.6)."

**What we take:** the principled form of "stop destroying resolution". A faithful HRNet is a large
rewrite **and is not recurrent** — we would have to decide where the ConvLSTM lives. Candidate (5) is
therefore explicitly *HRNet-lite*: one parallel full-resolution stream, one fusion. **We do not claim to
be testing HRNet.**

### `Islam2020_PositionEncoding` → why CoordConv failed here, and a caution

> "CNNs implicitly encode absolute position information in their feature maps… **Zero-padding is the
> primary source**. Removing all zero-padding from VGG16 reduces horizontal-pattern SPC from 0.742 to
> 0.381."

> "Position information is **not uniformly distributed across depth**: deeper layers encode
> progressively more… VGG16 deepest block 0.657 SPC vs shallowest 0.101."

> "Networks trained on **position-dependent tasks** encode substantially more… VGG-SS 0.982 vs 0.742."

**What we take:** an explanation for our own CoordConv negative — position was already implicit, so
explicit coordinates were redundant. All our convs use `padding=1`, so the mechanism is present.
**Caution for candidate (4):** if deeper layers carry more position information, removing depth/pooling
may *cost* positional encoding even as it preserves resolution. (4) is not unambiguously good.

### `Liu2018_CoordConv` → the closed question, recorded so it is not reopened

> "CoordConv… solves the coordinate transform problem with perfect generalization… **With coordinate
> weights set to zero, CoordConv is mathematically equivalent to ordinary convolution.**"

> "For ImageNet classification — a task requiring straightforward translation invariance — CoordConv
> shows **no statistically significant improvement**."

**What we take:** CoordConv helps where the task *is* a coordinate transform. Ours is not, and our own
3-seed × 300-lesson test found it **hurts** occurrence at every horizon. **Not re-tested.**

### `Radford2022_HighResConflictConvLSTM` → our own domain, and a defect we had not considered

> "The ConvLSTM predicts **de-escalation far better than escalation**: for escalation cell-months the
> model predicted equal or greater magnitude only **0.7%** of the time, vs **43.8%** for de-escalation."

> "The ConvLSTM assumes an **equal-area grid** but PRIO-GRID cells are degrees-based, causing areas to
> vary from **2,124 to 3,077 km²**. The learned spatial convolutions therefore represent different
> spatial areas for different regions, and simply **controlling for cell land area as an additional
> feature will not correct this**, because the convolutional filters themselves are applied uniformly."

**What we take:** the first quote independently reproduces our timid-body finding in the same domain.
The second is a **spatial-precision defect baked into our data representation** that no candidate here
addresses — a 3×3 kernel spans ~45% more area at one end of the grid than the other. **Out of scope for
this program; recorded as the strongest candidate for the next one.**

## Held, read, not load-bearing here

`Ronneberger2015_UNet` (the ancestor), `Alom2018_R2UNet` (recurrent residual U-Net — closest structural
relative, worth mining if (5) wins), `Kohl2018_ProbabilisticUNet`, `Oktay2018_AttentionUNet` (learned
spatial gating of skips — a natural successor to (3) if FiLM wins), `Chen2017_DeepLabv3` /
`Chen2018_DeepLabv3Plus` / `Chen2019_AugmentedASPPRemoteSensing` / `Chen2024_ImprovedDeepLabv3PlusRS`
(atrous — the dilation in (4)), `Lin2017_FeaturePyramidNetworks`, `Wang2020_DeepHRNet`,
`Yu2021_LiteHRNet`, `Dumoulin2016_ConvolutionArithmetic` (the arithmetic reference for (4)),
`Tancik2020_FourierFeatures`, `Irvin2025_STPyramidFlow`, `Hatamizadeh2022_UNETR`/`SwinUNETR`,
`Ding2025_GCAResUNet`.

## Gaps to fetch

None blocking. If (1) wins, the anti-aliasing literature has successors worth holding; if (5) wins,
`Wang2020_DeepHRNet` and `Yu2021_LiteHRNet` are already on the shelf.
