import cv2
import numpy as np

def create_video(duration_seconds=10, fps=30):
    # Video settings
    width, height = 640, 480
    video_path = 'created_video.mp4'

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    rectangle_width = width // 2
    rectangle_height = height // 2

    for second in range(duration_seconds * fps):
        # Create a white frame
        frame = np.ones((height, width, 3), np.uint8) * 255

        # Every 2 seconds, add a rectangle half the size of the previous one
        if second > 0 and second % (2 * fps) == 0:
            rectangle_width //= 2
            rectangle_height //= 2

        rectangle_start = (width // 2 - rectangle_width // 2, height // 2 - rectangle_height // 2)
        rectangle_end = (width // 2 + rectangle_width // 2, height // 2 + rectangle_height // 2)
        frame = cv2.rectangle(frame, rectangle_start, rectangle_end, (0, 0, 0), -1)

        # Write the frame to the video file
        out.write(frame)

    # Close the video file
    out.release()

    print(f"Created video: {video_path}")

def find_differences(video_path, threshold=100000):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    _, prev_frame = cap.read()

    # Create a list to store difference information
    differences = []

    # Loop through the video
    second = 0
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # Break the loop when the end of the video is reached
        if not ret:
            break

        # Calculate the difference between two frames
        diff = cv2.absdiff(prev_frame, frame)

        # Calculate the total difference (sum of differences in color channels for color images)
        total_diff = np.sum(diff)

        # If the total difference exceeds the threshold, mark it as a difference
        if total_diff > threshold:
            differences.append(second)  # Add the time of difference in seconds

        # Set the current frame as the previous frame for the next iteration
        prev_frame = frame

        # Update the video time information
        second += 1 / cap.get(cv2.CAP_PROP_FPS)

    # Close the video file
    cap.release()

    return differences

# Create a video
create_video()

# Find difference seconds in the video
difference_seconds = find_differences('created_video.mp4')

# Print the found difference seconds
print(f"Difference seconds in the video: {difference_seconds}")
