from numpy.core.numeric import full
import pandas as pd
import ffmpeg
import os


def load_labels(path) -> pd.DataFrame:
    print("Loading labels at ", path)
    labels = pd.read_csv(path)
    return labels


def get_label_fps(labels):
    """
    Calculates the fps per label. This way we can fight class imbalance.

    Args:
        labels (Pandas.DataFrame): [description]

    Returns:
        Dict<string, number>: the fps per label
    """
    label_count = {}
    total_label_count = 0

    label_sum = labels.groupby("label")[["label", "duration"]].sum()

    for row in label_sum.itertuples():
        label = row.Index
        duration = row.duration
        label_count[label] = duration
        total_label_count += duration

    print(label_count)
    print(total_label_count)

    return label_count


def main(label_file, base_video_path, output_folder):
    labels = load_labels(label_file)
    # First calculate class distribution
    labels["duration"] = labels["end_frame"] - labels["begin_frame"]
    label_fps = get_label_fps(labels)

    # Base graph.
    ffmpeg_video_files = []

    # Take the unique video_files
    for file in labels["video_file"].unique():
        print("Starting with file:", file)
        full_video_file = os.path.join(base_video_path, file)
        ffmpeg_input = ffmpeg.input(full_video_file)
        trim_inputs = []
        for row in labels[labels["video_file"]==file].itertuples():
            fps = label_fps[row.label]
            start_frame = row.begin_frame
            end_frame = row.end_frame
            print(row, fps)
            ffmpeg_row_input = ffmpeg_input.trim(start_frame=start_frame, end_frame=end_frame).filter('fps', fps=f'1/{fps}')
            trim_inputs.append(ffmpeg_row_input)

        # We're done with this video, append it to the graph.
        ffmpeg_video_files.append(ffmpeg.concat(trim_inputs))

    # Create the final graph.
    # TODO: create a labels file that saves the label for each output.
    ffmpeg.concat(ffmpeg_video_files).output(f"{output_folder}/frame-%d.jpg", start_number=0).overwrite_output().run()

    return

    for row in labels.itertuples():
        print(row)
        begin_frame = row.begin_frame
        end_frame = row.end_frame
        file = row.video_file
        label = row.label
        stream = ffmpeg.input(os.path.join(base_video_path, file))

if __name__ == "__main__":
    main("data/trains.csv", "data/videos")
