import os
import random
import shutil

# Define the source folder and the destination folder
source_folder = "Gesture_Plots_Output\Session_1"
destination_folder = "Session_1_Split"

# Define the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through DATA_FOREARM and DATA_WRIST
for data_type in ["DATA_FOREARM", "DATA_WRIST"]:
    data_type_path = os.path.join(source_folder, data_type)
    gestures = os.listdir(data_type_path)

    for gesture in gestures:
        gesture_path = os.path.join(data_type_path, gesture)
        files = os.listdir(gesture_path)

        # Shuffle the files
        random.shuffle(files)

        # Calculate split indices
        total_files = len(files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        # Split the files
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Create train, val, and test folders
        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_folder = os.path.join(destination_folder, data_type, split, gesture)
            os.makedirs(split_folder, exist_ok=True)

            # Move the files to the respective folders
            for file in split_files:
                src_file = os.path.join(gesture_path, file)
                dst_file = os.path.join(split_folder, file)
                shutil.copy(src_file, dst_file)  # Use shutil.copy to copy files instead of moving

print("Dataset split completed!")