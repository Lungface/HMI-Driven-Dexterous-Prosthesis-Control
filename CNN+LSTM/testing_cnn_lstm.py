import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

# -------------------------------------------------
# CONFIG (match your training config)
# -------------------------------------------------
DATA_ROOT = r"C:\Users\laibi\Desktop\FYP\1.1.0fullsections"
MODALITY = "combined"
GESTURES = list(range(1, 17))
TARGET_LEN = 10240
RANDOM_SEED = random.seed()

MODEL_PATH = "emg_cnn_bilstm_combined.h5"  # path to your saved model

# -------------------------------------------------
# Utilities (same as in training code)
# -------------------------------------------------

def pad_or_crop(x, target_len):
    """
    x: (T, C)
    If T >= target_len: center-crop to target_len
    If T <  target_len: pad zeros at the end to target_len
    """
    T, C = x.shape
    if T == target_len:
        return x
    if T > target_len:
        start = (T - target_len) // 2
        end = start + target_len
        return x[start:end, :]
    else:
        pad_len = target_len - T
        pad = np.zeros((pad_len, C), dtype=x.dtype)
        return np.vstack([x, pad])


def load_random_participant_session3(
    data_root,
    modality="combined",
    gestures=None,
    target_len=10240,
    n_trials=20,
):
    """
    From Session3_no_split:
      data_root/Session3_no_split/Session3/<modality>/gesture_YY/participant_ZZ/trial_N.npy

    Randomly:
      - chooses one participant
      - picks up to n_trials from that participant (across all gestures)

    Returns:
      X: (N, target_len, C)
      y: (N,) labels in [0..15]
      participant_id: string like "participant_03"
      trial_info: list of (gesture_id, trial_path) for each sample
    """
    if gestures is None:
        gestures = list(range(1, 17))

    s3_modality_dir = Path(data_root) / "Session3_no_split" / "Session3" / modality
    if not s3_modality_dir.is_dir():
        raise RuntimeError(f"Session3 directory not found: {s3_modality_dir}")

    # Collect all participant names from all gestures
    participant_names = set()
    for g in gestures:
        gesture_dir = s3_modality_dir / f"gesture_{g:02d}"
        if not gesture_dir.is_dir():
            continue
        for p_dir in gesture_dir.glob("participant_*"):
            participant_names.add(p_dir.name)

    participant_names = sorted(participant_names)
    if not participant_names:
        raise RuntimeError("No participants found in Session3.")

    # Randomly choose one participant
    participant_id = random.choice(participant_names)
    print(f"Randomly chosen participant from Session3: {participant_id}")

    # Collect all trials for that participant across gestures
    trial_paths = []
    for g in gestures:
        gesture_dir = s3_modality_dir / f"gesture_{g:02d}"
        p_dir = gesture_dir / participant_id
        if not p_dir.is_dir():
            continue
        for trial_path in p_dir.glob("trial_*.npy"):
            trial_paths.append((g, trial_path))  # store (gesture_id, path)

    if not trial_paths:
        raise RuntimeError(f"No trials found for participant {participant_id} in Session3.")

    # Pick up to n_trials randomly
    if len(trial_paths) > n_trials:
        trial_paths = random.sample(trial_paths, n_trials)

    X_list = []
    y_list = []
    trial_info = []

    for g, trial_path in trial_paths:
        arr = np.load(trial_path)  # (T, C)
        arr = pad_or_crop(arr, target_len)
        label_idx = g - 1  # gestures 1..16 -> labels 0..15
        X_list.append(arr)
        y_list.append(label_idx)
        trial_info.append((g, trial_path))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print(f"Loaded {X.shape[0]} trials for {participant_id} from Session3.")
    return X, y, participant_id, trial_info


# -------------------------------------------------
# MAIN: evaluate random Session3 participant
# -------------------------------------------------

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # 1) Load normalization statistics saved during training
    norm_stats_path = Path(DATA_ROOT) / f"normalization_stats_{MODALITY}.npz"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(
            f"Normalization stats file not found: {norm_stats_path}\n"
            "Make sure you ran the training script that saves this file."
        )

    norm_data = np.load(norm_stats_path)
    train_mean = norm_data["train_mean"]
    train_std = norm_data["train_std"]
    print(f"Loaded normalization stats from: {norm_stats_path}")

    # 2) Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # 3) Load random Session3 participant (up to 20 trials)
    X_s3, y_s3, participant_id, trial_info = load_random_participant_session3(
        data_root=DATA_ROOT,
        modality=MODALITY,
        gestures=GESTURES,
        target_len=TARGET_LEN,
        n_trials=20,
    )

    # 4) Normalize using training statistics
    X_s3_norm = (X_s3 - train_mean) / train_std

    # 5) Run inference
    probs = model.predict(X_s3_norm)
    preds = np.argmax(probs, axis=1)

    # 6) Show results per trial
    correct = preds == y_s3
    for i, ((g, trial_path), y_true, y_pred, is_corr) in enumerate(
        zip(trial_info, y_s3, preds, correct)
    ):
        print(
            f"{i:02d} | file: {trial_path.name} | "
            f"gesture_true={g:02d} (label {y_true}) | "
            f"gesture_pred={y_pred + 1:02d} (label {y_pred}) | "
            f"{'CORRECT' if is_corr else 'WRONG'}"
        )

    acc = correct.mean()
    print(f"\nParticipant: {participant_id}")
    print(f"Trials evaluated: {len(y_s3)}")
    print(f"Accuracy on these trials: {acc * 100:.2f}%")