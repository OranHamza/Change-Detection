import cv2
import numpy as np

# Set the size of the images (you can adjust this as needed)
image_size = (500, 500)

# Set the number of images in the sequence
num_images = 10

# Set the amplitude of the random noise
noise_amplitude = 0.05

# Function to calculate the change magnitude in the region within a rectangle
def calculate_change_magnitude(image, start_point, end_point):
    region = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    return np.sum(region)

# Create a list to store frames with changes
frames_with_changes = []

# Create and save the images with random noise and a gradually appearing black rectangle
for i in range(num_images):
    # Create a white image using NumPy
    white_image = np.ones((image_size[1], image_size[0]), dtype=np.float32)
    
    # Add random noise with small amplitude
    random_noise = np.random.uniform(low=-noise_amplitude, high=noise_amplitude, size=white_image.shape)
    noisy_image = np.clip(white_image + random_noise, 0, 1)
    
    # Gradually increase the size of the black rectangle from frames 4 to 6
    if 4 <= i <= 6:
        start_point = (image_size[0] // (i + 1), image_size[1] // (i + 1))
        end_point = (3 * image_size[0] // (i + 1), 3 * image_size[1] // (i + 1))
        color = (0, 0, 0)  # Black color
        cv2.rectangle(noisy_image, start_point, end_point, color, -1)
        
        # Add frames with changes to the list
        frames_with_changes.append(i)
    
    # Save the image with a unique filename (e.g., noisy_white_0.png, noisy_white_1.png, ...)
    filename = f"noisy_white_{i}.png"
    cv2.imwrite(filename, (noisy_image * 255).astype(np.uint8))

    print(f"Image {i + 1} saved as {filename}")

# Analyze differences between consecutive frames
for i in range(1, num_images):
    # Load consecutive frames
    previous_frame = cv2.imread(f"noisy_white_{i-1}.png", cv2.IMREAD_GRAYSCALE)
    current_frame = cv2.imread(f"noisy_white_{i}.png", cv2.IMREAD_GRAYSCALE)

    # Calculate the change magnitude in the region between frames
    start_point = (image_size[0] // 4, image_size[1] // 4)
    end_point = (3 * image_size[0] // 4, 3 * image_size[1] // 4)

    change_magnitude = calculate_change_magnitude(current_frame, start_point, end_point)

    # Print the frame index and change magnitude
    print(f"Frame {i}: Change Magnitude = {change_magnitude}")

    # Print and display frames with changes
    if i in frames_with_changes:
        print(f"   Frame {i} contains a different rectangle.")
        cv2.imshow(f"Frame with Change {i}", current_frame)
        cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()
