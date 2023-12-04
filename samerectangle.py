import cv2
import numpy as np

def display_all_frames_in_one_window(frames):
    for frame in frames:
        cv2.imshow("All Frames", frame)
        cv2.waitKey(500)  # Wait for 500 milliseconds between frames

    cv2.destroyAllWindows()

def generate_sequence(num_frames, image_size, noise_amplitude):
    sequence = []
    
    for i in range(num_frames):
        # Create a white image using NumPy
        white_image = np.ones((image_size[1], image_size[0]), dtype=np.float32)
        
        # Add random noise with small amplitude
        random_noise = np.random.uniform(low=-noise_amplitude, high=noise_amplitude, size=white_image.shape)
        noisy_image = np.clip(white_image + random_noise, 0, 1)
        
        # Gradually increase the size of the black rectangle from frames 4 to 6
        if 4 <= i <= 6:
            start_point = (image_size[0] // 4, image_size[1] // 4)
            end_point = (3 * image_size[0] // 4, 3 * image_size[1] // 4)
            alpha = (i - 4) / 2.0  # Linear interpolation factor
            color = (0, 0, 0)  # Black color
            cv2.rectangle(noisy_image, start_point, end_point, color, -1)
        
        sequence.append(noisy_image)
    
    return sequence

def calculate_change_magnitude(image, start_point, end_point):
    region = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    return np.sum(region)

if __name__ == "__main__":
    # Set the size of the images (you can adjust this as needed)
    image_size = (500, 500)

    # Set the number of images in the sequence
    num_images = 10

    # Set the amplitude of the random noise
    noise_amplitude = 0.05

    # Generate the sequence using the new function
    my_frames = generate_sequence(num_images, image_size, noise_amplitude)

    # Display all frames in one window
    display_all_frames_in_one_window(my_frames)

    # Analyze differences between frames
    for i in range(1, num_images):
        # Load consecutive frames
        previous_frame = my_frames[i-1] * 255
        current_frame = my_frames[i] * 255

        # Compute absolute difference between frames
        frame_difference = cv2.absdiff(previous_frame.astype(np.uint8), current_frame.astype(np.uint8))

        # Set a threshold for the difference
        threshold = 50
        _, thresholded_difference = cv2.threshold(frame_difference, threshold, 255, cv2.THRESH_BINARY)

        # Count non-zero pixels to determine the magnitude of the change
        change_magnitude = np.count_nonzero(thresholded_difference)

        # Print the frame index and change magnitude
        print(f"Frame {i}: Change Magnitude = {change_magnitude}")

        # Check if a change is detected
        if change_magnitude > 0:
            print(f"   Change detected on Frame {i}")
            cv2.imshow(f"Change Detected on Frame {i}", current_frame)
            cv2.waitKey(0)

    # Pencereleri kapat
    cv2.destroyAllWindows()
