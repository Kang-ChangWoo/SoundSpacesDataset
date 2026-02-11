# Detailed Report: Spherical Harmonics Integration for Audio-to-ERP Depth Estimation

## 1. Motivation and Theoretical Background

### 1.1 The Task: Audio Spectrogram -> ERP Depth Map

Our pipeline takes **binaural audio spectrograms** (2 channels) as input and predicts **ERP (equirectangular projection) depth maps** (256x512) as output.

```
Input:  Audio spectrogram  (B, 2, 256, 512)   [binaural stereo]
Output: ERP depth map      (B, 1, 256, 512)   [depth on the sphere]
```

The output ERP depth map is a projection of the full sphere onto a rectangular grid:

```
ERP pixel (i, j) --> spherical coords (theta, phi)
  theta (colatitude) = pi * i / H      (0 to pi,   top to bottom)
  phi   (azimuth)    = 2*pi * j / W    (0 to 2*pi, left to right)
```

### 1.2 The Problem with Standard UNet

The original `unet_256` treats the output ERP depth map as a flat 2D image. However, since the output represents depth on the **full sphere**, standard 2D convolutions in the decoder have no awareness of:
- The spherical geometry of the output space
- Distortion near the poles (top/bottom) where ERP stretches the sphere
- Global geometric consistency across the full 360-degree view

### 1.3 Why Spherical Harmonics Help

**Spherical Harmonics (SH)** are the natural orthonormal basis for functions defined on the sphere -- analogous to Fourier basis for functions on a line.

**Key distinction -- two components**:
- **SH basis functions** (fixed, precomputed): These are mathematical functions defined on the sphere. They provide a structured representation of the output depth space.
- **SH coefficients** (learned, audio-conditioned): These are predicted **from the encoded audio features** at the bottleneck. The encoder processes the audio spectrogram, and the SH branch takes those encoded audio features to predict 55 coefficients.

The final SH map = `einsum(audio-predicted coefficients, fixed SH basis)`. This means the SH map is **fully conditioned on the audio input** -- it is NOT a static geometric prior.

| SH Order (l) | What it captures | Analogy |
|---|---|---|
| l=0 | Constant (average depth) | DC component |
| l=1 | Linear gradient (floor/ceiling tilt) | Low frequency |
| l=2-3 | Room shape (walls, corners) | Mid frequency |
| l=4-9 | Furniture, objects, fine structure | Higher frequency |

By decomposing the depth prediction into SH coefficients predicted from audio features, the network gains:
1. **Global spherical awareness** -- SH basis naturally encodes the relationship between all points on the sphere
2. **Structured regularization** -- Low-order SH prevents noisy, inconsistent depth predictions
3. **Audio-conditioned geometry** -- The encoder learns to map audio features (reverberation, echoes) to SH coefficients that represent room-level depth structure

### 1.4 Audio + SH Synergy

Audio is particularly well-suited for SH because:
- **Reverberation** in audio encodes global room geometry (size, shape) -- this maps directly to low-order SH
- **Binaural cues** (interaural time/level differences) encode directional information -- this maps to SH orientation
- **Echo patterns** encode distance to surfaces at different directions -- this is essentially what SH coefficients represent

### 1.5 Reference: HUSH Paper

The implementation is adapted from **HUSH** ([github.com/vision3d-lab/HUSH](https://github.com/vision3d-lab/HUSH)), which uses:
- Swin Transformer encoder + Deformable Attention decoder
- SH Coefficient Extractor from decoder features
- SH-based cross-attention module for depth/normal prediction
- Degree 10 SH (55 basis functions)

We adapt the core SH concept to work with the existing UNet architecture for audio-to-depth.

---

## 2. Architecture Design

### 2.1 Overall Architecture: `UnetSHGenerator`

```
Audio Spectrogram (B, 2, 256, 512)  [binaural stereo]
         |
    ============
    |  ENCODER  |   8 downsampling levels (Conv 4x4, stride 2)
    ============    Learns audio features: reverberation, echoes,
         |          binaural cues, spectral patterns
  Bottleneck (B, 512, 1, 2)   <-- encoded audio features
    /          \
   |            |
   v            v
============  ================================
| DECODER  |  | SH BRANCH                     |
| 7 levels |  | (uses SAME encoded audio feat) |
| + skips  |  | GlobalAvgPool -> (B,512)       |
| (full    |  | MLP: 512 -> 256 -> 55          |
|  512ch)  |  | (severe compression: 512->55)  |
============  ================================
   |               |
   v               v
Features       SH Coefficients (B, 55)
(B,128,128,256)    |  [audio-predicted, learned]
   |               v
   |          einsum(coeffs, SH_basis)
   |            [learned]  [fixed]
   |               |
   |               v
   |          SH Map (B, 1, 256, 512)
   |          [audio-conditioned spherical depth map]
   |               |
   +-------+-------+
           |
    ===============
    | SH FUSION   |  (mid-level, 256ch)
    | (additive)  |  features + gate * sh_feat
    ===============
           |
    Decoder trans3
           |
    ===============
    | SH FUSION   |  (late, 128ch)
    | (additive)  |  features + gate * sh_feat
    ===============
           |
    Outer Decoder
    ConvTranspose -> Sigmoid
           |
    ERP Depth (B, 1, 256, 512)
```

**Data flow explanation**:
1. Audio spectrogram encodes spatial audio information (echoes, reverberation, binaural cues)
2. Encoder extracts hierarchical audio features down to a compact bottleneck
3. Bottleneck captures global scene understanding from audio
4. **SH branch** takes the **encoded audio features from the bottleneck** and maps them into 55 SH coefficients via MLP -- predicting "how deep is the scene in each spherical direction" based on what the encoder learned from the audio
5. SH coefficients (audio-conditioned) are projected onto pre-computed SH basis functions (fixed) to produce an audio-conditioned spherical depth map
6. **SH fusion** gates the decoder features using this audio-conditioned SH map, enhancing spherically-consistent regions
7. Decoder produces the final high-resolution depth map

**Important**: Both the decoder AND the SH branch receive information from the same bottleneck (encoded audio features). The SH branch is an auxiliary path that forces the bottleneck to also learn a compact spherical representation of the scene, which is then re-injected into the decoder.

### 2.2 Key Difference from Original UNet

The original `UnetGenerator` is **recursive** (each block wraps its inner block via `nn.Sequential`), making it impossible to access intermediate features. The new `UnetSHGenerator` is **flat** (explicit encoder/decoder lists), which:
- Is functionally identical to the original (same layers, same channel counts, same skip connections)
- Allows extracting the bottleneck features for SH coefficient prediction
- Enables injecting SH information into the decoder path

### 2.3 Encoder Path (unchanged behavior)

| Level | Input Channels | Output Channels | Resolution (H x W) |
|-------|---------------|----------------|-------------------|
| enc0  | 2 (stereo audio) | 64          | 256x512 -> 128x256 |
| enc1  | 64            | 128            | 128x256 -> 64x128  |
| enc2  | 128           | 256            | 64x128 -> 32x64    |
| enc3  | 256           | 512            | 32x64 -> 16x32     |
| enc4  | 512           | 512            | 16x32 -> 8x16      |
| enc5  | 512           | 512            | 8x16 -> 4x8        |
| enc6  | 512           | 512            | 4x8 -> 2x4         |
| enc_inner | 512       | 512            | 2x4 -> 1x2 (bottleneck) |

### 2.4 SH Branch (new)

**Step 1: SH Basis Construction** (`build_real_sh_basis`)

Pre-computed at initialization using `scipy.special.sph_harm`:
- Degree = 10, giving **55 basis functions** (l=0..9, m=0..l for each l)
- Each basis function is a 256x512 map of real SH values on the ERP grid
- Stored as a non-trainable buffer: shape `(55, 256, 512)`
- Uses HUSH convention: only non-negative m (real part), so N = degree*(degree+1)/2

```python
# For each pixel (i,j) in the ERP output grid, compute Y_l^m(theta, phi):
theta = pi * i / H     # colatitude
phi   = 2*pi * j / W   # azimuth
Y_real = sqrt(2) * Re(Y_l^m(theta, phi))  for m > 0
Y_real = Y_l^0(theta, phi)                 for m = 0
```

**Step 2: SH Coefficient Extraction** (`SHCoeffExtractor`) -- **uses encoded audio features**

```
Bottleneck (B, 512, 1, 2)     <-- these ARE the encoded audio features from the encoder
    -> AdaptiveAvgPool2d(1)    -> (B, 512, 1, 1)
    -> Flatten                 -> (B, 512)
    -> Linear(512, 256) + ReLU -> (B, 256)
    -> Linear(256, 55)         -> (B, 55)    [SH coefficients, predicted from audio]
```

The SH coefficients are **not fixed or precomputed** -- they are learned predictions from the audio-encoded bottleneck features. The network learns to map audio features to SH coefficients. For example:
- Audio with strong reverberation -> larger room -> larger low-order SH coefficients
- Binaural differences at certain frequencies -> wall direction -> specific SH mode activations

**Note on information bottleneck**: The bottleneck has 512 channels, but this is compressed to just 55 scalar SH coefficients, then expanded to a single-channel SH map. This is a severe dimensionality reduction (512 channels -> 55 scalars -> 1 channel), which limits how much audio information can pass through the SH branch.

**Step 3: SH Map Reconstruction**

```python
sh_map = einsum('bn, nhw -> bhw', sh_coeffs, sh_basis)
# (B, 55) x (55, 256, 512) -> (B, 256, 512)
#  ^audio-predicted   ^fixed precomputed
```

This produces a full-resolution **audio-conditioned spherical depth map** -- a weighted combination of fixed SH basis functions, where the weights (coefficients) come from the encoded audio features. The result is a smooth, globally-consistent depth estimate based on room geometry inferred from audio. However, it is limited to the expressiveness of 55 SH basis functions (degree 10), so it can only capture smooth/low-frequency depth structure.

**Step 4: SH Fusion** (`SHFusionModule`) -- **Multi-level, additive**

SH fusion is applied at **two decoder levels** for stronger influence:

**Mid-level fusion** (256ch = 128+128 after skip with enc1):
```
Decoder features (B, 256, 64, 128)
SH map (B, 1, 256, 512)
    -> Bilinear resize to (B, 1, 64, 128)
    -> Conv2d(1, 64, 3x3) + BN + ReLU     -> sh_feat (B, 64, 64, 128)
    -> Concat [features, sh_feat]          -> (B, 320, 64, 128)
    -> Conv2d(320, 256, 1x1) + Sigmoid     -> gate (B, 256, 64, 128)
    -> features + gate * sh_feat           -> fused features
```

**Late fusion** (128ch = 64+64 after skip with enc0):
```
Decoder features (B, 128, 128, 256)
SH map (B, 1, 256, 512)
    -> Bilinear resize to (B, 1, 128, 256)
    -> Conv2d(1, 64, 3x3) + BN + ReLU     -> sh_feat (B, 64, 128, 256)
    -> Concat [features, sh_feat]          -> (B, 192, 128, 256)
    -> Conv2d(192, 128, 1x1) + Sigmoid     -> gate (B, 128, 128, 256)
    -> features + gate * sh_feat           -> fused features
```

The additive formula `features + gate * sh_feat` means:
- SH feature information is directly added to the decoder features, weighted by a learned gate
- The gradient signal through the additive path is stronger than the multiplicative path
- The network can still learn gate close to 0 to reduce SH influence, but cannot entirely bypass it without gradient penalty

### 2.5 Decoder Path (unchanged behavior + SH fusion)

| Level | Input Channels | Output Channels | Skip From |
|-------|---------------|----------------|-----------|
| dec_inner | 512       | 512            | -         |
| dec6  | 1024 (512+512)| 512            | enc6      |
| dec5  | 1024          | 512            | enc5      |
| dec4  | 1024          | 512            | enc4      |
| dec3  | 1024          | 256            | enc3      |
| dec2  | 512           | 128            | enc2      |
| dec1  | 256           | 64             | enc1      |
| **SH fusion (mid)** applied here (256ch = 128+128 after skip with enc1) | | |
| dec1 (trans3) | 256 | 64 | enc1 |
| **SH fusion (late)** applied here (128ch = 64+64 after skip with enc0) | | |
| dec_outer | 128       | 1              | enc0      |

---

## 3. Loss Functions

### 3.1 BerHu Loss (Reverse Huber)

Replaces the original L1 loss. Defined piecewise:

```
              |  |e|                          if |e| <= delta
BerHu(e) =   |
              |  (e^2 + delta^2) / (2*delta)  if |e| > delta

where delta = 0.2 * max(|target - pred|)   (adaptive per batch)
```

**Why BerHu > L1 for depth:**

| Property | L1 | BerHu |
|----------|-----|-------|
| Small errors | Linear penalty | Linear penalty (same sharpness) |
| Large errors | Linear penalty | Quadratic penalty (stronger correction) |
| Threshold | N/A | Adaptive per batch (20% of max error) |
| Edge preservation | Good | Good (L1 region handles edges) |
| Outlier handling | Moderate | Strong (L2 pushes large errors down faster) |

### 3.2 Gradient Loss

Penalizes differences in spatial gradients between predicted and ground truth depth:

```python
gx = conv2d(pred - target, [[ 0,-1, 0],[ 0, 0, 0],[ 0, 1, 0]])  # vertical gradient
gy = conv2d(pred - target, [[ 0, 0, 0],[-1, 0, 1],[ 0, 0, 0]])  # horizontal gradient
loss = mean(|gx| + |gy|)
```

This ensures **edge sharpness** -- depth discontinuities at object boundaries are preserved rather than blurred.

### 3.3 SH Auxiliary Loss

Directly supervises the SH map against a smoothed version of the ground truth depth:

```
SH_Aux_Loss = L1(sh_map, AvgPool_then_upsample(gt_depth))   [on valid pixels]
```

SH can only represent low-frequency structure, so the GT is smoothed with average pooling (16x16 kernel) before comparison. This forces the SH branch to learn meaningful depth coefficients rather than relying solely on indirect gradient signal from the final depth loss.

### 3.4 Combined Training Loss

```
Total Loss = BerHu(pred, gt) + lambda_grad * GradientLoss(pred, gt) + lambda_sh * SH_Aux_Loss(sh_map, gt)
```

Defaults: `lambda_grad = 0.5`, `lambda_sh = 0.1`. This combination targets:
- BerHu: overall depth accuracy (absolute values)
- Gradient: structural accuracy (edges and transitions)
- SH Aux: direct supervision of the SH branch for meaningful spherical depth coefficients

---

## 4. Parameter and Memory Analysis

```
                        Original UNet-256    UNet-256-SH
---------------------------------------------------------
Total parameters:          54,409,857          54,581,368
  Encoder:                 19,538,240          19,538,240  (identical)
  Decoder:                 34,871,617          34,872,193  (identical*)
  SH Coeff Extractor:              -             145,463
  SH Fusion Module:                 -              25,472
---------------------------------------------------------
Overhead:                           -     +171,511 (0.32%)
SH Basis buffer:                    -      7,208,960 floats (27.5 MB)
Total param memory:          207.6 MB            208.2 MB
```

*The decoder has a trivially different count because the flat implementation uses separate `nn.ModuleList` instead of nested `nn.Sequential`, but the architecture is functionally identical.

**Key takeaway**: Only **0.32% additional trainable parameters**, but this also means the SH branch has very limited capacity:
- SH basis is precomputed (non-trainable buffer) -- fixed mathematical functions, not learned
- SH coefficient extraction is just global pooling + 2 linear layers -- compresses 512 channels to 55 scalars
- SH fusion is just 2 convolution layers at a single injection point
- The decoder's full 512-channel path is vastly richer than the SH branch's 1-channel output

---

## 5. Performance Analysis

### 5.1 How the SH Branch Uses Audio Features

To be explicit about the data flow:

```
Audio Spectrogram (input)
    |
    v
Encoder (8 levels of convolutions)          <-- learns audio features
    |
    v
Bottleneck (B, 512, 1, 2)                   <-- encoded audio representation
    |
    +---> Decoder (skip connections)         <-- uses encoded audio features (full 512 channels)
    |
    +---> SH Branch:                         <-- ALSO uses encoded audio features, but:
          GlobalAvgPool -> (B, 512)               compresses 512 channels to 55 scalars
          MLP -> (B, 55)                          then expands to 1-channel map
          einsum(55 coeffs, 55 basis) -> (B,1,256,512)
```

**Both paths use the same encoded audio features from the bottleneck.** The decoder gets the full 512-channel representation, while the SH branch compresses it to 55 scalar coefficients and reconstructs a single-channel spherical depth map.

### 5.2 Why Performance Did Not Improve -- Analysis and Fixes

After the initial `train_erp_sh.sh` run, no measurable improvement was observed. Six issues were identified, three of which have now been resolved:

| Issue | Status | Explanation |
|-------|--------|-------------|
| **Information redundancy** | Fundamental | The SH branch and the decoder both read from the same bottleneck. The decoder already has access to all the information the SH branch uses -- the SH path adds no new information, just a different representation of the same features |
| **Severe information bottleneck** | Fundamental | 512 channels -> 55 scalars -> 1 channel map. The SH branch discards most of the encoded audio information. The decoder's 512-channel path is far richer |
| **~~Late fusion, single injection point~~** | **RESOLVED** | ~~SH map was fused only once, at the second-to-last decoder layer.~~ Now fused at **two** levels: mid-level (256ch) and late (128ch), so SH influences earlier decoder stages |
| **~~Residual gating = easy to ignore~~** | **RESOLVED** | ~~The `features * (1 + gate)` design let the network learn `gate ≈ 0`.~~ Now uses **additive fusion** (`features + gate * sh_feat`), providing a stronger gradient signal through the SH path |
| **SH basis may not match audio** | Fundamental | HUSH used SH with a Swin Transformer + Deformable Attention on **RGB images** (dense visual features). Audio spectrograms contain much sparser spatial information -- the encoder may not learn bottleneck features that meaningfully decompose into SH coefficients |
| **~~No direct SH supervision~~** | **RESOLVED** | ~~SH coefficients were only supervised via the final depth loss.~~ Now an **SH auxiliary loss** directly supervises the SH map against a smoothed version of the ground truth depth, forcing the SH branch to learn meaningful coefficients |

### 5.3 Loss Function Improvements

| Mechanism | Effect |
|-----------|--------|
| **BerHu adaptive threshold** | Automatically balances precision (small errors) and recall (large errors) per batch, adapting to the difficulty of each batch |
| **Gradient loss** | Directly optimizes edge quality, addressing the common UNet problem of over-smoothed depth boundaries |
| **Combined loss** | Multi-objective training prevents the network from collapsing to average-depth predictions |

### 5.4 Metric Expectations (Revised)

| Metric | Original Expectation | Actual Observation | Likely Reason |
|--------|---------------------|-------------------|---------------|
| **RMSE** | Lower | No improvement | SH branch adds redundant information; decoder already captures this |
| **ABS_REL** | Lower | No improvement | 55 SH coefficients too few to meaningfully represent complex indoor depth |
| **Delta1** | Higher | No improvement | SH fusion too late in the pipeline to affect edge quality |
| **MAE** | Lower | No improvement | Network likely learns to ignore the SH branch (gate ≈ 0) |

### 5.5 Implemented Improvements

| Approach | Status | Details |
|----------|--------|---------|
| **SH auxiliary loss** | **Implemented** | Directly supervises the SH map against a smoothed (avg-pooled) GT depth, forcing the SH branch to learn meaningful coefficients. Weight: 0.1 |
| **Multi-level SH fusion** | **Implemented** | SH information injected at two decoder levels (mid: 256ch, late: 128ch) instead of one |
| **Additive fusion** | **Implemented** | Replaced `features * (1 + gate)` with `features + gate * sh_feat` for stronger gradient signal |

### 5.6 Remaining Potential Directions

| Approach | Rationale |
|----------|-----------|
| **Separate SH decoder head** | Instead of gating, use SH map + decoder features as two independent predictions and combine them (e.g., learned weighted average) |
| **Higher SH degree** | Increase from degree 10 (55 coeffs) to degree 16+ for finer detail |
| **Spherical convolutions instead of SH** | Replace standard 2D convolutions in the decoder with spherical convolutions (e.g., SphereNet, SpherePHD) that natively handle ERP distortion |
| **Focus on loss improvements only** | BerHu + Gradient loss may provide more benefit than the SH branch -- run ablation without SH to isolate their effect |

---

## 6. Implementation Details

### 6.1 SH Basis Verification

The implementation was verified to produce correct SH values:
- DC component (l=0, m=0) has mean = 0.2821, matching the theoretical value of 1/sqrt(4*pi) = 0.2821
- Non-DC components have mean approximately 0 (orthogonality property)
- Basis shape: (55, 256, 512) for degree=10

### 6.2 Forward Pass Verification

- Input: (B, 2, 256, 512) audio spectrogram (binaural stereo)
- Output: (B, 1, 256, 512) ERP depth map in [0, 1] (when depth_norm=True)
- Backward pass confirmed: all parameters receive gradients
- Deterministic in eval mode: same input produces same output

### 6.3 Compatibility

- Works with audio spectrogram input (2 channels) -- the primary use case
- Also works with RGB input (3 channels) if needed
- Compatible with existing checkpoint loading/saving mechanism
- All evaluation metrics (RMSE, ABS_REL, Delta1/2/3, MAE, Log10) unchanged
- TensorBoard logging works identically

---

## 7. File Change Summary

| File | Status | Description |
|------|--------|-------------|
| `models/sh_utils.py` | **NEW** | SH basis computation, coefficient extractor, fusion module with additive fusion (130 lines) |
| `models/unet_sh_model.py` | **NEW** | `UnetSHGenerator` - flat UNet with multi-level SH fusion, returns `(depth, sh_map)` tuple |
| `train_erp_sh.sh` | **NEW** | Training script for audio->ERP depth with SH + BerHu + Gradient + SH aux loss |
| `models/unetbaseline_model.py` | MODIFIED | Added `unet_256_sh` / `unet_128_sh` to `define_G` |
| `utils_criterion.py` | MODIFIED | Added `BerHuLoss`, `GradientLoss`, and `SHAuxiliaryLoss` classes |
| `train.py` | MODIFIED | Support for BerHu, gradient loss, SH aux loss, and tuple model output |
| `eval.py` | MODIFIED | Support for BerHu criterion, handles tuple model output |
| `test.py` | MODIFIED | Support for BerHu criterion, handles tuple model output |
| `conf/mode/train.yaml` | MODIFIED | Added `use_grad_loss`, `grad_loss_weight`, `use_sh_aux_loss`, `sh_aux_loss_weight` options |
| `conf/model/unet_baseline.yaml` | MODIFIED | Added `sh_degree`, documented new generators |

---

## 8. Usage

### 8.1 Training with All Improvements

```bash
# Using the provided script (recommended)
# Now includes: multi-level additive SH fusion + SH auxiliary loss
bash train_erp_sh.sh

# Or manually:
python3 train.py \
    model.generator=unet_256_sh \
    model.sh_degree=10 \
    mode.criterion=BerHu \
    mode.use_grad_loss=True \
    mode.grad_loss_weight=0.5 \
    mode.use_sh_aux_loss=True \
    mode.sh_aux_loss_weight=0.1 \
    dataset.input_type=audio \
    dataset.audio_format=spectrogram \
    dataset.depth_type=erp \
    dataset.depth_norm=True \
    'dataset.images_size=[256,512]' \
    mode.batch_size=16 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    mode.epochs=100
```

### 8.2 Ablation Configurations (Recommended to Isolate Effects)

Since the SH branch did not improve performance, these ablations help identify which component (if any) provides benefit:

```bash
# Ablation 1: SH model + L1 loss (isolate SH contribution)
python3 train.py model.generator=unet_256_sh mode.criterion=L1 \
    dataset.input_type=audio dataset.depth_type=erp 'dataset.images_size=[256,512]'

# Ablation 2: Standard UNet + BerHu + Gradient loss (isolate loss contribution, no SH)
# This is the most important ablation -- loss improvements may help even without SH
python3 train.py model.generator=unet_256 mode.criterion=BerHu mode.use_grad_loss=True \
    dataset.input_type=audio dataset.depth_type=erp 'dataset.images_size=[256,512]'

# Ablation 3: Baseline (standard UNet + L1 loss, no SH, no loss improvements)
python3 train.py model.generator=unet_256 mode.criterion=L1 \
    dataset.input_type=audio dataset.depth_type=erp 'dataset.images_size=[256,512]'
```

### 8.3 Hyperparameter Tuning

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `model.sh_degree` | 10 | 4-16 | Higher = more SH basis functions, finer geometric detail |
| `mode.grad_loss_weight` | 0.5 | 0.1-2.0 | Higher = sharper edges but potential instability |
| `mode.sh_aux_loss_weight` | 0.1 | 0.01-0.5 | Weight for SH auxiliary loss; higher = stronger SH supervision |
| BerHu threshold | 0.2 | 0.1-0.5 | Higher = more L1-like; lower = more L2-like |

---

## 9. Dependencies

- **scipy** (already installed) - for `scipy.special.sph_harm` SH basis computation
- **PyTorch** (already installed) - core framework
- No new package installations required (`torch_harmonics` from HUSH was replaced with scipy)
