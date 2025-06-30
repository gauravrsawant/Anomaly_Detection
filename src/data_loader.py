import os
import numpy as np
import cv2 as cv
import gc

def preprocess_training_clips(filepath, img_size, frames_per_clip=10):
    all_clips = []

    # Iterate over each subfolder (each video sequence)
    for folder in sorted(os.listdir(filepath)):
        folder_path = os.path.join(filepath, folder)

        if os.path.isdir(folder_path):
            frames = []

            # Read and preprocess each .tif image in the folder
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.tif'):
                    image_path = os.path.join(folder_path, file)

                    # Read in grayscale, resize, normalize
                    img = cv.imread(image_path, 0)
                    img = cv.resize(img, (img_size, img_size), interpolation=cv.INTER_CUBIC)
                    img = img / 255.0
                    frames.append(img)

            # Create clips using non-overlapping windows of frames_per_clip
            num_frames = len(frames)
            for start in range(0, num_frames - frames_per_clip + 1, frames_per_clip):
                clip = np.zeros((frames_per_clip, img_size, img_size, 1))
                for k in range(frames_per_clip):
                    clip[k, :, :, 0] = frames[start + k]
                all_clips.append(clip)

            # Cleanup memory
            del frames
            gc.collect()

    return np.array(all_clips)

if __name__ == "__main__":
    TRAIN_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
    IMG_SIZE = 256
    training_data = preprocess_training_clips(TRAIN_DIR, IMG_SIZE)
    print(f"Training data shape: {training_data.shape}")
