
import cv2
import os

# Function to split video into frames
def split_video_to_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"Total frames extracted: {frame_count}")

# Function to perform blob detection
def detect_blobs(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Set up the blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 150  # Adjust as needed
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs
    keypoints = detector.detect(gray_image)
    return keypoints

# Function to save blobs as images
def save_blobs_as_images(frame_path, keypoints, output_dir):
    image = cv2.imread(frame_path)
    for i, kp in enumerate(keypoints):
        x = int(kp.pt[0] - kp.size / 2)
        y = int(kp.pt[1] - kp.size / 2)
        w = int(kp.size)
        h = int(kp.size)
        blob_roi = image[y:y+h, x:x+w]
        if blob_roi.size > 0:
            blob_path = os.path.join(output_dir, f"{os.path.basename(frame_path).split('.')[0]}_blob_{i:02d}.jpg")
            cv2.imwrite(blob_path, blob_roi)

# Paths
video_path_1 = "/mnt/data/video4.avi"
video_path_2 = "/mnt/data/video5.avi"
frames_output_dir = "/mnt/data/frames"
blobs_output_dir = "/mnt/data/blobs"

# Create output directories if they don't exist
os.makedirs(frames_output_dir, exist_ok=True)
os.makedirs(blobs_output_dir, exist_ok=True)

# Split videos into frames
split_video_to_frames(video_path_1, frames_output_dir)
split_video_to_frames(video_path_2, frames_output_dir)

# Process each frame for blob detection and save blobs
for frame_file in os.listdir(frames_output_dir):
    frame_path = os.path.join(frames_output_dir, frame_file)
    frame_image = cv2.imread(frame_path)
    keypoints = detect_blobs(frame_image)
    save_blobs_as_images(frame_path, keypoints, blobs_output_dir)

print("Processing completed.")