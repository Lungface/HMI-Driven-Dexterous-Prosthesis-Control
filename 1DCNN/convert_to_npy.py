import os
import sys
import shutil

import numpy as np
import wfdb


# ------------------------------ configuration ------------------------------ #

SESSIONS = [1, 2, 3]
N_GESTURES = 16               # active gestures
N_TRIALS = 7                  # trials per gesture
N_GESTURES_WITH_REST = 17     # 16 gestures + 1 rest (original MATLAB uses 17)
OUTPUT_ROOT = "Output BM NPY"     # keep same name as MATLAB version

# Channel mapping as in the MATLAB code (assumes 32 physical channels)
# First 16 channels = forearm, some of the remaining = wrist.
forearm_channels_mask = np.concatenate(
    (np.ones(8), np.ones(8), np.zeros(8), np.zeros(8))
).astype(bool)

wrist_channels_mask = np.concatenate(
    (np.zeros(8), np.zeros(8), np.zeros(1),
     np.ones(6), np.zeros(2),
     np.ones(6), np.zeros(1))
).astype(bool)


# ------------------------------ utilities ---------------------------------- #

def ask_overwrite_folder(path: str) -> None:
    """Create `path` or ask user whether to overwrite it if it exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        return

    while True:
        print(f"Found existing folder: {path}")
        answer = input("Overwrite it (Y/N)? ").strip().upper()
        if answer in ("Y", "N"):
            break

    if answer == "N":
        print("Exiting script without changes.")
        sys.exit(0)

    print("Overwriting existing folder...")
    shutil.rmtree(path)
    os.makedirs(path)


def get_number_of_subjects(session1_path: str) -> int:
    """
    Infer number of participants from Session1 folder.
    Assumes participant folders are named: session1_participant{index}
    """
    entries = os.listdir(session1_path)
    subs = [
        d for d in entries
        if os.path.isdir(os.path.join(session1_path, d))
        and d.startswith("session1_participant")
    ]
    if not subs:
        raise RuntimeError(f"No participant folders found in {session1_path}")
    return len(subs)


# ------------------------------ main script -------------------------------- #

def main():
    base_dir = os.getcwd()

    # Check that Session1, Session2, Session3 exist
    for s in SESSIONS:
        session_dir = os.path.join(base_dir, f"Session{s}")
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"Missing folder: {session_dir}")

    # Determine number of participants from Session1
    session1_path = os.path.join(base_dir, "Session1")
    nsub = get_number_of_subjects(session1_path)
    print(f"Detected {nsub} participants in Session1.")

    # Prepare output root
    ask_overwrite_folder(os.path.join(base_dir, OUTPUT_ROOT))

    total_files = len(SESSIONS) * nsub
    processed_files = 0

    for isession in SESSIONS:
        session_input_root = os.path.join(base_dir, f"Session{isession}")
        session_output_root = os.path.join(base_dir, OUTPUT_ROOT, f"Session{isession}_converted")
        os.makedirs(session_output_root, exist_ok=True)

        for isub in range(1, nsub + 1):
            foldername = f"session{isession}_participant{isub}"
            participant_input_folder = os.path.join(session_input_root, foldername)

            if not os.path.isdir(participant_input_folder):
                raise FileNotFoundError(f"Missing participant folder: {participant_input_folder}")

            # --- Read first record to discover nsamples and n_channels --- #
            example_filepath = os.path.join(
                participant_input_folder,
                f"{foldername}_gesture1_trial1"
            )

            if not (os.path.isfile(example_filepath + ".dat") and
                    os.path.isfile(example_filepath + ".hea")):
                raise FileNotFoundError(f"Example WFDB files not found: {example_filepath}.dat/.hea")

            example_record = wfdb.rdrecord(example_filepath)
            nsamples = example_record.sig_len
            n_channels = example_record.p_signal.shape[1]

            # Sanity check: channel masks must match the number of channels
            if n_channels != len(forearm_channels_mask) or n_channels != len(wrist_channels_mask):
                raise ValueError(
                    f"Channel count mismatch: record has {n_channels} channels, "
                    f"but masks expect {len(forearm_channels_mask)}"
                )

            # Number of forearm/wrist channels actually used
            n_forearm = int(forearm_channels_mask.sum())
            n_wrist = int(wrist_channels_mask.sum())

            print(
                f"Session {isession}, Participant {isub}: "
                f"{nsamples} samples, {n_channels} channels "
                f"({n_forearm} forearm, {n_wrist} wrist)."
            )

            # --- Allocate arrays: (trial, gesture, samples, channels) --- #
            DATA_FOREARM = np.zeros(
                (N_TRIALS, N_GESTURES_WITH_REST, nsamples, n_forearm),
                dtype=np.float64,
            )
            DATA_WRIST = np.zeros(
                (N_TRIALS, N_GESTURES_WITH_REST, nsamples, n_wrist),
                dtype=np.float64,
            )

            # --- Fill arrays with data from all gestures and trials --- #
            for igesture in range(1, N_GESTURES_WITH_REST + 1):  # 1..17
                for itrial in range(1, N_TRIALS + 1):             # 1..7
                    filename = f"{foldername}_gesture{igesture}_trial{itrial}"
                    filepath = os.path.join(participant_input_folder, filename)

                    if not (os.path.isfile(filepath + ".dat") and
                            os.path.isfile(filepath + ".hea")):
                        raise FileNotFoundError(f"Missing WFDB files: {filepath}.dat/.hea")

                    record = wfdb.rdrecord(filepath)
                    data_emg = record.p_signal  # shape (nsamples, n_channels)

                    if data_emg.shape[0] != nsamples:
                        raise ValueError(
                            f"Inconsistent sample length in {filename}: "
                            f"expected {nsamples}, got {data_emg.shape[0]}"
                        )

                    # Split into forearm and wrist using masks
                    data_forearm = data_emg[:, forearm_channels_mask]
                    data_wrist = data_emg[:, wrist_channels_mask]

                    # Check shapes once
                    if data_forearm.shape[1] != n_forearm or data_wrist.shape[1] != n_wrist:
                        raise ValueError(
                            f"Unexpected channel count in {filename}: "
                            f"forearm {data_forearm.shape[1]}, wrist {data_wrist.shape[1]}"
                        )

                    DATA_FOREARM[itrial - 1, igesture - 1, :, :] = data_forearm
                    DATA_WRIST[itrial - 1, igesture - 1, :, :] = data_wrist

            # --- Save participant data for this session as .npz --- #
            output_path = os.path.join(session_output_root, f"{foldername}.npz")
            np.savez_compressed(
                output_path,
                DATA_FOREARM=DATA_FOREARM,
                DATA_WRIST=DATA_WRIST,
            )

            processed_files += 1
            print(f"Saved {output_path} ({processed_files}/{total_files})")

    print("All sessions and participants converted successfully.")


if __name__ == "__main__":
    main()