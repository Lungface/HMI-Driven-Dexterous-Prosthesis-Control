import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from concurrent.futures import ProcessPoolExecutor


base_directory = 'Output BM'
output_directory = 'Gesture_Plots_Output'


def process_one_mat_file(session, input_directory, mat_file):
    """Process a single .mat file: load, loop over datasets, trials, gestures, channels, and save plots."""
    mat_file_path = os.path.join(input_directory, mat_file)

    # Load the .mat file
    mat_data = loadmat(mat_file_path)

    # Process both DATA_FOREARM and DATA_WRIST
    for data_key in ['DATA_FOREARM', 'DATA_WRIST']:
        if data_key not in mat_data:
            print(f"{data_key} not found in {mat_file}. Skipping...")
            continue

        data = mat_data[data_key]
        print(f"[Session {session}] Processing file: {mat_file}, Dataset: {data_key}")
        print("Num Trials:", data.shape[0], "Num Gestures:", data.shape[1])

        # Loop through all trials and gestures
        for trial in range(data.shape[0]):
            for gesture in range(data.shape[1]):
                data_to_plot = data[trial, gesture]
                print(f"File {mat_file} - Trial {trial + 1}, Gesture {gesture + 1}")
                print("Timesteps:", data_to_plot.shape[0], "Num Channels:", data_to_plot.shape[1])

                # Loop through all channels
                for ichannel in range(data_to_plot.shape[1]):
                    time = range(data_to_plot.shape[0])

                    # Create a folder for the current gesture if it doesn't exist
                    gesture_folder = os.path.join(
                        output_directory,
                        f"Session_{session}",
                        data_key,
                        f"Gesture_{gesture + 1}"
                    )
                    os.makedirs(gesture_folder, exist_ok=True)

                    # Create and save the plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(time, data_to_plot[:, ichannel])
                    plt.title(
                        f'File: {mat_file}, Dataset: {data_key}, '
                        f'Trial {trial + 1}, Gesture {gesture + 1}, Channel {ichannel + 1}'
                    )
                    plt.xlabel('Time')
                    plt.ylabel('Amplitude')
                    plt.grid(True)

                    plot_filename = (
                        f"{os.path.splitext(mat_file)[0]}_"
                        f"Trial{trial + 1}_Gesture{gesture + 1}_Channel{ichannel + 1}.png"
                    )
                    plot_filepath = os.path.join(gesture_folder, plot_filename)
                    plt.savefig(plot_filepath)
                    plt.close()


if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Collect all jobs (each job is one .mat file)
    jobs = []
    for session in range(1, 2):  # Sessions 1 (adjust range as needed)
        input_directory = os.path.join(base_directory, f'Session{session}_converted')

        if not os.path.exists(input_directory):
            print(f"Session directory {input_directory} does not exist. Skipping...")
            continue

        for mat_file in os.listdir(input_directory):
            if mat_file.endswith('.mat'):
                jobs.append((session, input_directory, mat_file))

    # Run the jobs in parallel using multiple processes
    # Set max_workers to the number of CPU cores or adjust manually
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_one_mat_file, session, input_dir, mat_file)
            for (session, input_dir, mat_file) in jobs
        ]
        # Optionally wait for all to complete and re-raise exceptions
        for f in futures:
            f.result()