from numpy.core.numeric import full
import pandas as pd
import ffmpeg
import os


def load_labels(path) -> pd.DataFrame:
    print("Loading labels at ", path)
    labels = pd.read_csv(path, usecols=["label","begin_frame","end_frame","video_file"])
    return labels


def add_idle(labels: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(labels)
    for row in labels.itertuples():
        row_dict = dict(row._asdict())
        half_duration = round(0.5 * row.duration)
        del row_dict["Index"]
        row_dict["label"] = "empty"
        row_dict["begin_frame"] = row.begin_frame - half_duration
        row_dict["end_frame"] = row.begin_frame - 1
        row_dict["duration"] = half_duration
        result = result.append(row_dict, ignore_index=True)
        row_dict["begin_frame"] = row.end_frame + 1
        row_dict["end_frame"] = row.end_frame + half_duration
        row_dict["duration"] = half_duration
        result = result.append(row_dict, ignore_index=True)
    return result


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


def frame_number_to_time(frame_number, frame_rate):
    format = f"{round(frame_number / frame_rate) // 3600:02d}:{(round(frame_number / frame_rate) // 60) % 60:02d}:{(frame_number / frame_rate) % 60}"
    return format


def main(label_file, base_video_path, output_folder):
    labels = load_labels(label_file)
    # First calculate class distribution
    labels["duration"] = labels["end_frame"] - labels["begin_frame"]
    labels = add_idle(labels)
    label_fps = get_label_fps(labels)

    output_index = pd.DataFrame(columns=['file', 'class'])

    # Take the unique video_files
    current_image_index = 0
    for file in labels["video_file"].unique():
        video_file = file
        if not video_file.endswith(".mp4"):
            video_file = video_file + ".mp4"
        full_video_file = os.path.join(base_video_path, video_file)
        print("Starting with file:", full_video_file)
        for row in labels[labels["video_file"]==file].itertuples():
            # TODO: calculate the fps based on this.
            # fps = label_fps[row.label]
            fps = 10
            input_fps = 30
            start_frame = row.begin_frame
            end_frame = row.end_frame
            start_frame_time = frame_number_to_time(start_frame + 60, input_fps)
            vframes = round((end_frame-start_frame)//(input_fps/fps))
            print(f"Start time: {start_frame_time}, vframes: {vframes}")

            output_file_format = os.path.join(output_folder, "frame-%d.jpg")
            tmp_command = ffmpeg \
                .input(full_video_file, ss=(start_frame_time)) \
                .filter('fps', fps=fps, round='up') \
                .output(output_file_format, vframes=vframes)
            print(tmp_command.get_args())
            tmp_command.run()

            # Rename the files.
            for file_index in range(vframes):
                new_name =  f"output-frame-{current_image_index + file_index}.jpg"
                os.replace(os.path.join(os.getcwd(), output_folder, f"frame-{file_index + 1}.jpg"), os.path.join(os.getcwd(), output_folder, new_name))

                output_index = output_index.append({"file": new_name, "class": row.label}, ignore_index=True)

            current_image_index += vframes
            # return

        print("Finished file: ", full_video_file)

    output_index.to_parquet(os.path.join(output_folder, "label_index.parquet"))


if __name__ == "__main__":
    print(pd.read_parquet("data/output/label_index.parquet"))
    # main("data/Labels.csv", "data/videos", "data/output")
