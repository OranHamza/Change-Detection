import cv2
import numpy as np

def create_video(duration_seconds=10, fps=30):
    width, height = 640, 480
    video_path = 'created_video.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    rectangle_width = width // 2
    rectangle_height = height // 2

    for second in range(duration_seconds * fps):
        frame = np.ones((height, width, 3), np.uint8) * 255

        if second > 0 and second % (2 * fps) == 0:
            rectangle_width //= 2
            rectangle_height //= 2

        rectangle_start = (width // 2 - rectangle_width // 2, height // 2 - rectangle_height // 2)
        rectangle_end = (width // 2 + rectangle_width // 2, height // 2 + rectangle_height // 2)
        frame = cv2.rectangle(frame, rectangle_start, rectangle_end, (0, 0, 0), -1)

        out.write(frame)

    out.release()

    print(f"Created video: {video_path}")

def find_differences(video_path, threshold=100000):
    cap = cv2.VideoCapture(video_path)
    _, prev_frame = cap.read()
    differences = []
    second = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        diff = cv2.absdiff(prev_frame, frame)
        total_diff = np.sum(diff)
        if total_diff > threshold:
            differences.append(second)
        prev_frame = frame
        second += 1 / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return differences

create_video()
difference_seconds = find_differences('created_video.mp4')
print(f"Difference seconds in the video: {difference_seconds}")
