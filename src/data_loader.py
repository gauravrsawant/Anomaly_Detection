def preprocess_clips(filepath, img_size, frames_per_clip=10, filter_with_gt=False):
    clips = []
    for folder in sorted(os.listdir(filepath)):
        folder_path = os.path.join(filepath, folder)
        if not os.path.isdir(folder_path):
            continue
        if filter_with_gt and not os.path.exists(os.path.join(filepath, folder + "_gt")):
            continue

        frames = []
        for file in sorted(os.listdir(folder_path)):
            if file.endswith('.tif'):
                img = cv.imread(os.path.join(folder_path, file), 0)
                img = cv.resize(img, (img_size, img_size)) / 255.0
                frames.append(img)

        for start in range(0, len(frames) - frames_per_clip + 1, frames_per_clip):
            clip = np.zeros((frames_per_clip, img_size, img_size, 1))
            for k in range(frames_per_clip):
                clip[k, :, :, 0] = frames[start + k]
            clips.append(clip)
        del frames
        gc.collect()

    return np.array(clips)


def generate_frame_labels(gt_base_dir):
    labels = []
    for folder in sorted(os.listdir(gt_base_dir)):
        if not folder.endswith("_gt"):
            continue
        folder_path = os.path.join(gt_base_dir, folder)
        for bmp in sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')]):
            mask = cv.imread(os.path.join(folder_path, bmp), 0)
            labels.append(1 if np.any(mask > 0) else 0)
    return np.array(labels)