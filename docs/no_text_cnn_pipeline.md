# Legacy No-Text CNN Pipeline

This note is grounded in the reference files you provided:

- `loop_cnn/model.py`
- `loop_cnn/dataset.py`
- `loop_cnn/train.py`
- `loop_cnn/drive.py`
- `client/cnn_loop_session.py`
- `client/episode_manager.py`

It describes the older no-language path-following stack, not the newer language-conditioned version.

## A. Model Architecture

The CNN model in `loop_cnn/model.py` is a small regression network that predicts normalized body commands:

- input: stacked recent RGB frames
- output: normalized `[vx, vy, omega]`
- output range: `[-1, 1]` on each axis via a final `tanh`

### Input shape

`LoopPolicyConfig` defines:

- `image_width`
- `image_height`
- `frame_history`
- `action_dim = 3`

The effective input channel count is:

```text
input_channels = frame_history * 3
```

So with the default history of 3 frames, the model input is:

```text
[B, 9, H, W]
```

That comes directly from concatenating three RGB frames along the channel dimension.

### Encoder

The encoder is four convolution blocks followed by global pooling:

1. `Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)`
2. `Conv2d(32, 64, kernel_size=3, stride=2, padding=1)`
3. `Conv2d(64, 128, kernel_size=3, stride=2, padding=1)`
4. `Conv2d(128, 128, kernel_size=3, stride=2, padding=1)`
5. `AdaptiveAvgPool2d((1, 1))`

Each conv block is `Conv -> BatchNorm -> ReLU`.

### Head

The MLP head is:

1. `Flatten`
2. `Linear(128, hidden_dim)`
3. `ReLU`
4. `Dropout`
5. `Linear(hidden_dim, 32)`
6. `ReLU`
7. `Linear(32, 3)`
8. `Tanh`

With the default config:

- `hidden_dim = 64`
- `dropout = 0.1`

### What the model predicts

The model predicts the current normalized control action:

```text
[vx, vy, omega]
```

Semantically, those are body-frame commands, not wheel velocities.

## B. Dataset Format

### Folder structure

The CNN dataset loader in `loop_cnn/dataset.py` scans for:

```text
episodes_root/
  session_name/
    episode_00000/
      video.mp4
      data.parquet
      episode_info.json
```

The loader discovers episodes with `discover_cnn_episodes()` by checking for exactly those three files.

### Episode files

The key files are:

- `video.mp4`: RGB video frames for the episode
- `data.parquet`: per-frame metadata and actions
- `episode_info.json`: episode-level metadata, including direction

The loader uses:

- `data.parquet["action"]`
- `data.parquet["task"]`
- `episode_info.json["direction"]`

So those fields are the hard compatibility requirements.

### Per-frame data

From the session recorder code you provided, the intended semantics are:

- `state`: previous normalized action
- `action`: current normalized action
- `timestamp`: episode-relative seconds
- `task`: path-following task label
- `task_index`: task id

The training loader only consumes `action`, but the rest of the format matters for consistency and inspection.

### Video usage

`LoopEpisodeDataset` decodes `video.mp4` with PyAV and resizes frames to the model input size. It does not load images from loose files. The video is the image source of truth.

### Session split

Train/val splitting is session-based, not frame-based:

- `split_sessions()` groups by `session_name`
- a validation subset of sessions is held out
- this avoids leaking near-identical consecutive frames across splits

If there is only one session, validation becomes empty.

### Sampling and weighting

The dataset builds one sample per frame and computes sample weights from the action:

- very small actions get weight `0.35`
- turning or lateral corrections get weight `1.2`
- otherwise weight `0.8`

Then `train.py` uses `WeightedRandomSampler` with replacement. So the pipeline intentionally oversamples turning and correction behavior relative to idle or straight-driving frames.

## C. Training

### Training command

In this standalone repo, the intended training command is:

```bash
cd /workspace/turbopi_standalone
PYTHONPATH=/workspace/turbopi_standalone /workspace/isaaclab/_isaac_sim/python.sh -m cnn_policy.train --episodes-dir /workspace/turbopi_standalone/data/cnn_square_loop --run-dir /workspace/turbopi_standalone/runs/cnn_v1 --device cuda
```

The old `loop_cnn.train` module from the legacy repo is not needed here; the standalone repo now ships its own `cnn_policy.train`.

### Data loading

`train.py` calls `build_datasets()` from `loop_cnn/dataset.py`, which builds:

- a training dataset with augmentation enabled
- a validation dataset with augmentation disabled

If the dataset is small enough, it preloads decoded episodes into RAM to avoid repeated video decode.

### Augmentation and preprocessing

The training dataset can apply:

- brightness jitter
- contrast jitter
- saturation jitter
- hue jitter
- affine rotation
- translation
- optional Gaussian blur

Frames are resized, converted to tensors, and stacked channel-wise.

### Loss

Training uses `nn.HuberLoss(delta=args.huber_delta)`.

Validation uses the same Huber loss plus MAE tracked separately for:

- `vx`
- `vy`
- `omega`

### Optimization

The optimizer is `AdamW`.

The learning-rate schedule is `CosineAnnealingLR`.

### Validation behavior

If there is no validation split, the code warns and reuses train metrics as the comparison signal.

Each epoch writes:

- `checkpoints/last.pt`
- `checkpoints/best.pt`
- per-epoch checkpoint copies
- `training_summary.json`

The summary is updated even on `KeyboardInterrupt`.

## D. Inference

### Turning camera frames into a tensor

In this standalone repo, the runtime path is:

- load a checkpoint with `cnn_policy.drive.LoopPolicyRuntime`
- launch the Isaac Lab driver script at [`scripts/drive_turbopi_square_cnn.py`](../scripts/drive_turbopi_square_cnn.py)
- read the robot-mounted RGB camera each control tick

The frame processing itself still follows the same pattern as the legacy `loop_cnn/drive.py`:

1. the robot client returns an RGB frame
2. the frame is resized with PIL
3. `torchvision.transforms.functional.to_tensor()` converts it to `[3, H, W]`
4. recent frames are stacked into `[1, frame_history * 3, H, W]`

### Prediction

The model output is clipped to `[-1, 1]` and then smoothed:

```text
smoothed = alpha * previous_action + (1 - alpha) * predicted_action
```

That smoothing is an EMA on the normalized action.

### Denormalization

`denormalize_action()` multiplies normalized action by per-axis caps:

- `vx_cap`
- `vy_cap`
- `omega_cap`

So model output stays normalized, while deployment chooses the real command scale later.

### Deadzone handling

`apply_minimum_command_floor()` lifts small nonzero commands above motor deadzones:

- if the value is effectively zero, it is forced to zero
- if it is nonzero but below the configured minimum floor, it is bumped up to that floor with the same sign

This is important because small predictions can otherwise disappear into drivetrain deadzones.

### What can go wrong

From the code path you shared, common deployment failure modes are:

- camera feed lag or dropouts
- aggressive smoothing causing sluggish turns
- too-small command caps causing understeer
- too-large command caps causing oscillation
- minimum command floors that are too high, which can make the robot jerky
- domain mismatch between training track appearance and live deployment

## E. Deadlock / Freeze Issue in Isaac Lab Teleop

This is the issue from the previous Isaac Lab session, and it has two separate causes.

### 1. Timeline state

The original teleop/view scripts could look dead if the simulation timeline was not already in Play. In Isaac Lab, if the timeline is paused or stopped, your script can still be running while physics never advances.

That is why the standalone scripts were patched to:

- call `sim.play()` after reset
- call `sim.play()` again if the loop notices the sim is not currently playing

Without that, the script appears frozen even though Python is alive.

### 2. Viewport focus

For keyboard teleop over Brev or livestream, keyboard events only reach Isaac Lab after the browser session gives focus to the viewport. So the script can also look dead even when the timeline is running, simply because the viewport never received the key events.

That is why keyboard teleop needs one click in the viewport first.

### Why automated scripts should avoid both traps

An automated recorder should:

- never depend on keyboard focus
- explicitly start or resume the timeline
- use bounded timeouts
- treat paused simulation as recoverable, not as a condition to wait on forever

That is exactly why the new standalone recorder is autonomous rather than piggybacking on the teleop path.
