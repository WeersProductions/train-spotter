from numpy.core.numeric import full
import pandas as pd
import ffmpeg
import os


def load_labels(path) -> pd.DataFrame:
    print("Loading labels at ", path)
    labels = pd.read_csv(path)
    return labels


def main(label_file, base_video_path):
    labels = load_labels(label_file)
    # First calculate class distribution
    label_count = {}
    total_label_count = 0
    labels["duration"] = labels["end_frame"] - labels["begin_frame"]
    label_sum = labels.groupby("label")[["label", "duration"]].sum()

    for row in label_sum.itertuples():
        label = row.Index
        duration = row.duration
        label_count[label] = duration
        total_label_count += duration
        print(label, duration)
    return
    # Take the unique video_files
    for file in labels["video_file"].unique():
        print("Starting with file:", file)
        full_file = os.path.join(base_video_path, file)
        trim_inputs = []


    for row in labels.itertuples():
        print(row)
        begin_frame = row.begin_frame
        end_frame = row.end_frame
        file = row.video_file
        label = row.label
        stream = ffmpeg.input(os.path.join(base_video_path, file))

if __name__ == "__main__":
    main("data/trains.csv", "data/videos")
