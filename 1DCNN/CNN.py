import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------- CONFIG ---------------------- #

# Folder where the filtered, split data is stored
DATA_ROOT = r"C:\Users\laibi\Desktop\FYP\1.1.0fullsections\Output_filtered"

SESSIONS = [1, 2]            # which sessions to use

# Choose which data to use: "forearm", "wrist", or "combined"
MODALITY = "combined"        # <- set this to "combined" to use 28-channel data

GESTURES = list(range(1, 17))  # use gesture 1..16

# Typical trial length is 10240 samples
TARGET_LEN = 10240           # timesteps (pad/crop each trial to this length)

BATCH_SIZE = 32
EPOCHS = 30
RANDOM_SEED = 42


# ---------------------- UTILITIES ---------------------- #

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


def load_trials_from_split(
    data_root,
    split,          # "train", "val", or "test"
    sessions,
    modality="combined",   # "forearm", "wrist", or "combined"
    gestures=None,
    target_len=10240,
):
    """
    Walks through:
      data_root/<split>/SessionX/<modality>/gesture_YY/participant_ZZ/trial_N.npy

    modality folder name must match how you saved the data:
      - "forearm"  -> forearm-only (16 ch)
      - "wrist"    -> wrist-only   (12 ch)
      - "combined" -> forearm+wrist (28 ch)

    Returns:
      X: (N, target_len, n_channels)
      y: (N,) gesture labels [0..n_classes-1]
    """
    if gestures is None:
        gestures = list(range(1, 17))

    X_list = []
    y_list = []

    split_root = Path(data_root) / split

    for s in sessions:
        session_dir = split_root / f"Session{s}" / modality
        if not session_dir.is_dir():
            print(f"[{split}] Warning: {session_dir} not found, skipping.")
            continue

        for g in gestures:
            gesture_dir = session_dir / f"gesture_{g:02d}"
            if not gesture_dir.is_dir():
                continue

            label_idx = g - 1  # gestures 1..16 -> labels 0..15

            for participant_dir in gesture_dir.glob("participant_*"):
                for trial_path in participant_dir.glob("trial_*.npy"):
                    arr = np.load(trial_path)  # (T, C)
                    arr = pad_or_crop(arr, target_len)
                    X_list.append(arr)
                    y_list.append(label_idx)

    if not X_list:
        raise RuntimeError(f"No trials found for split='{split}' under {data_root} (modality='{modality}')")

    X = np.stack(X_list, axis=0)  # (N, T, C)
    y = np.array(y_list, dtype=np.int64)

    print(f"[{split}] Loaded trials: {X.shape[0]}")
    print(f"[{split}] Input shape  : {X.shape[1:]} (T, C)")
    print(f"[{split}] Num classes  : {len(np.unique(y))}")

    return X, y


def build_cnn_model(input_length, n_channels, n_classes):
    """
    Simple 1D CNN over time.
    Input per sample: (T, C)
    """
    inputs = layers.Input(shape=(input_length, n_channels))

    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=7, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def plot_history(history):
    df = pd.DataFrame(history.history)
    df.index = df.index + 1  # epoch numbers start at 1 for nicer plots

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    df.plot(y=["loss", "val_loss"], grid=True, ax=axs[0])
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training and Validation Loss")

    # Accuracy: handle both 'accuracy' and 'acc'
    if "accuracy" in df.columns and "val_accuracy" in df.columns:
        df.plot(y=["accuracy", "val_accuracy"], grid=True, ax=axs[1])
    elif "acc" in df.columns and "val_acc" in df.columns:
        df.plot(y=["acc", "val_acc"], grid=True, ax=axs[1])
    else:
        axs[1].text(0.5, 0.5, "No accuracy metric found",
                    ha="center", va="center")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.show()

# ---------------------- MAIN ---------------------- #

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # 1. Load training trials from 'train' split
    X_train, y_train = load_trials_from_split(
        data_root=DATA_ROOT,
        split="train",
        sessions=SESSIONS,
        modality=MODALITY,
        gestures=GESTURES,
        target_len=TARGET_LEN,
    )

    # 2. Shuffle training trials (optional but recommended)
    N_train = X_train.shape[0]
    perm = np.random.permutation(N_train)
    X_train = X_train[perm]
    y_train = y_train[perm]

    print("After shuffle:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # 3. Load validation trials from 'val' split
    X_val, y_val = load_trials_from_split(
        data_root=DATA_ROOT,
        split="val",
        sessions=SESSIONS,
        modality=MODALITY,
        gestures=GESTURES,
        target_len=TARGET_LEN,
    )

    print("Val shape  :", X_val.shape, y_val.shape)

    # 4. Build model
    n_classes = len(np.unique(y_train))
    n_channels = X_train.shape[2]   # will be 28 for "combined"
    model = build_cnn_model(
        input_length=TARGET_LEN,
        n_channels=n_channels,
        n_classes=n_classes,
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",        # or "val_accuracy"
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )
    
    # 5. Train with validation set from disk
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early_stop, reduce_lr],
    )
    
    plot_history(history)

    # 6. Load test set from 'test' split (participants not seen in training/val)
    X_test, y_test = load_trials_from_split(
        data_root=DATA_ROOT,
        split="test",
        sessions=SESSIONS,
        modality=MODALITY,
        gestures=GESTURES,
        target_len=TARGET_LEN,
    )

    # 7. Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # 8. Save model
    model_name = f"emg_cnn_{MODALITY}.h5"
    model.save(model_name)
    print(f"Model saved as {model_name}")
