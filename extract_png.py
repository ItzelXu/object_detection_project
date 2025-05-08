import os
import cv2
def extract_frame(video_path, frame_index, output_path):
    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total_frames:
        print(f"Warning: {video_path} contains only {total_frames} frames, cannot extract frame {frame_index}")
        cap.release()
        return False

    # Set the video's current position to the desired frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_index} from {video_path}")
        cap.release()
        return False

    # Write the frame to disk as PNG.
    cv2.imwrite(output_path, frame)
    cap.release()
    print(f"Saved frame {frame_index} of '{os.path.basename(video_path)}' as '{os.path.basename(output_path)}'")
    return True


if __name__ == "__main__":
    # Folder containing the .mp4 videos
    video_dir = os.path.join("output", "video_inference")

    # Set which frame index you want from each video (0-indexed)
    frame_index = 100  # Adjust this value as needed

    # Iterate through all files in the video directory
    for file in os.listdir(video_dir):
        if file.lower().endswith(".mp4"):
            video_path = os.path.join(video_dir, file)
            base_name, _ = os.path.splitext(file)
            # Create an output filename with correlated naming, e.g. "video1_frame100.png"
            output_filename = f"{base_name}_frame{frame_index}.png"
            output_path = os.path.join(video_dir, output_filename)
            extract_frame(video_path, frame_index, output_path)
