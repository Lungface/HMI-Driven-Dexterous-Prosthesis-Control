import os
import re
import json
import random
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# ------------------- Filtering function ------------------- #

def emg_bandpass_notch_filter(
    data,
    fs=2048,
    bandpass=(20, 450),
    notch_freq=50,
    notch_q=30,
    use_notch=True,
):
    """
    data: (T, C) numpy array
    returns: (T, C) filtered
    """
    x = np.asarray(data, dtype=np.float64).copy()
    x = x - np.mean(x, axis=0, keepdims=True)

    nyq = 0.5 * fs
    low = bandpass[0] / nyq
    high = bandpass[1] / nyq
    b_bp, a_bp = butter(4, [low, high], btype="band")

    if use_notch:
        w0 = notch_freq / nyq
        b_notch, a_notch = iirnotch(w0, notch_q)
    else:
        b_notch = a_notch = None

    for ch in range(x.shape[1]):
        x[:, ch] = filtfilt(b_bp, a_bp, x[:, ch])
        if use_notch:
            x[:, ch] = filtfilt(b_notch, a_notch, x[:, ch])

    return x


# ------------- Process one participant's npz file ------------- #

def process_participant_npz(
    input_npz_path: Path,
    session_out_root: Path,
    fs=2048,
    bandpass=(20, 450),
    notch_freq=50,
    notch_q=30,
    use_notch=True,
):
    """
    - Load one .npz: DATA_FOREARM (trials, gestures, samples, 16),
                     DATA_WRIST   (trials, gestures, samples, 12)
    - Filter all trials for both forearm and wrist
    - Save per-trial combined arrays into structure under session_out_root:
        Session{s}/
          combined/gesture_XX/participant_YY/trial_T.npy
    """

    data = np.load(input_npz_path)
    DATA_FOREARM = data["DATA_FOREARM"]  # (n_trials, n_gestures, n_samples, 16)
    DATA_WRIST   = data["DATA_WRIST"]    # (n_trials, n_gestures, n_samples, 12)

    n_trials, n_gestures, n_samples, n_ch_forearm = DATA_FOREARM.shape
    _, _, _, n_ch_wrist = DATA_WRIST.shape

    print(f"Processing {input_npz_path.name}: "
          f"{n_trials} trials, {n_gestures} gestures, "
          f"{n_samples} samples, {n_ch_forearm} forearm ch, {n_ch_wrist} wrist ch")

    # Parse participant id from filename like "session1_participant3.npz"
    m = re.search(r"participant(\d+)", input_npz_path.stem)
    participant_id = int(m.group(1)) if m else 0

    # Only combined output root
    combined_root = session_out_root / "combined"
    combined_root.mkdir(parents=True, exist_ok=True)

    # Loop over gestures and trials
    for ig in range(n_gestures):      # 0..16 (assuming 17 gestures incl. rest)
        gesture_id = ig + 1
        for it in range(n_trials):    # 0..6 (7 trials)
            trial_id = it + 1

            # -------- Forearm (16 ch) --------
            seg_f = DATA_FOREARM[it, ig]  # (samples, 16)
            seg_f_filt = emg_bandpass_notch_filter(
                seg_f,
                fs=fs,
                bandpass=bandpass,
                notch_freq=notch_freq,
                notch_q=notch_q,
                use_notch=use_notch,
            ).astype(np.float32)

            # -------- Wrist (12 ch) --------
            seg_w = DATA_WRIST[it, ig]  # (samples, 12)
            seg_w_filt = emg_bandpass_notch_filter(
                seg_w,
                fs=fs,
                bandpass=bandpass,
                notch_freq=notch_freq,
                notch_q=notch_q,
                use_notch=use_notch,
            ).astype(np.float32)

            # -------- Combined (16 + 12 = 28 ch) --------
            if seg_f_filt.shape[0] != seg_w_filt.shape[0]:
                raise ValueError(
                    f"Sample length mismatch for participant {participant_id}, "
                    f"gesture {gesture_id}, trial {trial_id}: "
                    f"forearm={seg_f_filt.shape[0]}, wrist={seg_w_filt.shape[0]}"
                )

            seg_combined = np.concatenate([seg_f_filt, seg_w_filt], axis=1)  # (samples, 28)

            c_gesture_dir = combined_root / f"gesture_{gesture_id:02d}" / f"participant_{participant_id:02d}"
            c_gesture_dir.mkdir(parents=True, exist_ok=True)
            c_path = c_gesture_dir / f"trial_{trial_id}.npy"
            np.save(c_path, seg_combined)


# ---------------------- Helper: split participants ---------------------- #

def split_participants_3way(participants, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split a set/list of participant IDs into train/val/test sets.
    Default ratios: 80% train, 10% val, 10% test.
    """
    participants = sorted(participants)
    rng = random.Random(seed)
    rng.shuffle(participants)

    n_total = len(participants)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    # Remaining go to test
    n_test  = n_total - n_train - n_val

    train_participants = set(participants[:n_train])
    val_participants   = set(participants[n_train:n_train + n_val])
    test_participants  = set(participants[n_train + n_val:])

    assert len(train_participants) + len(val_participants) + len(test_participants) == n_total

    return train_participants, val_participants, test_participants


# ----------------------------- Main script ----------------------------- #

if __name__ == "__main__":
    base_dir = Path(os.getcwd())

    # Input root: where your original converted files are
    input_root = base_dir / "Output BM"

    # Output root: new structured, filtered, split data
    output_root = base_dir / "Output_filtered"

    # Only Sessions 1 and 2 (change if needed)
    sessions_to_process = [1, 2]

    # ------------------ 1) Collect all participant IDs ------------------ #
    participant_ids = set()
    for s in sessions_to_process:
        session_input_dir = input_root / f"Session{s}_converted"
        if not session_input_dir.is_dir():
            continue
        for npz_path in session_input_dir.glob("*.npz"):
            m = re.search(r"participant(\d+)", npz_path.stem)
            if m:
                participant_ids.add(int(m.group(1)))

    if not participant_ids:
        raise RuntimeError("No participant .npz files found under 'Output BM'.")

    # ------------------ 2) Split participants 80:10:10 ------------------ #
    train_participants, val_participants, test_participants = split_participants_3way(
        participant_ids, train_ratio=0.8, val_ratio=0.1, seed=42
    )

    print("All participants :", sorted(participant_ids))
    print("Train participants:", sorted(train_participants))
    print("Val participants  :", sorted(val_participants))
    print("Test participants :", sorted(test_participants))

    # Save split info
    output_root.mkdir(parents=True, exist_ok=True)
    split_info = {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "seed": 42,
        "participants_all": sorted(list(participant_ids)),
        "train_participants": sorted(list(train_participants)),
        "val_participants": sorted(list(val_participants)),
        "test_participants": sorted(list(test_participants)),
    }
    with open(output_root / "participant_split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # ------------------ 3) Process sessions with split ------------------ #
    for s in sessions_to_process:
        session_input_dir = input_root / f"Session{s}_converted"
        if not session_input_dir.is_dir():
            print(f"Warning: {session_input_dir} not found, skipping Session {s}.")
            continue

        npz_files = sorted(session_input_dir.glob("*.npz"))
        if not npz_files:
            print(f"No .npz files in {session_input_dir}, skipping Session {s}.")
            continue

        print(f"\n=== Session {s} ===")
        for npz_path in npz_files:
            m = re.search(r"participant(\d+)", npz_path.stem)
            if not m:
                print(f"Warning: cannot parse participant id from {npz_path.name}, skipping.")
                continue
            participant_id = int(m.group(1))

            # Decide split for this participant
            if participant_id in train_participants:
                split = "train"
            elif participant_id in val_participants:
                split = "val"
            else:
                split = "test"

            # Session output root for this split (contains combined/ subfolder)
            session_out_root = output_root / split / f"Session{s}"
            session_out_root.mkdir(parents=True, exist_ok=True)

            process_participant_npz(
                input_npz_path=npz_path,
                session_out_root=session_out_root,
                fs=2048,
                bandpass=(20, 450),
                notch_freq=50,
                notch_q=30,
                use_notch=True,
            )

    print("\nDone. Structured, filtered, split-by-participant data (combined only) saved to 'Output_filtered'.")