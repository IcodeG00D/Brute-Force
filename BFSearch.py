#BF Search Project
import cv2

# Define the paths of the target images
target_image_paths = [
'C:\\Users\\Kushagra pathak\\Desktop\\python\\testimg\\765f28e4-33d3-4a9c-be0e-4bc9e6cee4fe.jfif',
    'C:\\Users\\Kushagra pathak\\Desktop\\python\\testimg\\93c4ec94-5e2e-4e44-9baf-6f7f43eeb5c7.jfif',
    'C:\\Users\\Kushagra pathak\\Desktop\\python\\testimg\\d612eb6b-089c-4d91-b435-a03d51e01adf.jfif',
     # Add more image paths as needed
]

# Create a list to store the target images and their descriptors
target_images = []
target_keypoints = []
target_descriptors = []

# Create a feature detector
orb = cv2.ORB_create()

# Load the target images and compute descriptors
for path in target_image_paths:
    target_image = cv2.imread(path)
    keypoints_target, descriptors_target = orb.detectAndCompute(target_image, None)
    if descriptors_target is not None:
        target_images.append(target_image)
        target_keypoints.append(keypoints_target)
        target_descriptors.append(descriptors_target)

# Create a feature matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Detect features in the frame
    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame, None)

    best_match_idx = None  # Index of the best matching target image
    best_match_distance = float('inf')  # Initial distance set to infinity

    # Match the features between the frame and each target image
    for i, descriptors_target in enumerate(target_descriptors):
        matches = bf.match(descriptors_frame, descriptors_target)
        matches = sorted(matches, key=lambda x: x.distance)

        # Check if the current target image has a better match
        if matches[0].distance < best_match_distance:
            best_match_idx = i
            best_match_distance = matches[0].distance

    if best_match_idx is not None:
        # Draw the best match on the frame
        matched_frame = cv2.drawMatches(
            frame, keypoints_frame,
            target_images[best_match_idx], target_keypoints[best_match_idx], [matches[0]], None, flags=2
        )
        cv2.imshow('Object Detection', matched_frame)
    else:
        cv2.imshow('Object Detection', frame)

    # Check for the 'q' key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
