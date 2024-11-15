import cv2
import os
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to split video into frames
def split_video_to_frames(video_path, output_dir):
    if not os.path.exists(video_path):
        logging.error(f"Video file {video_path} does not exist.")
        return

    try:
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
        logging.info(f"Total frames extracted from {video_path}: {frame_count}")
    except Exception as e:
        logging.error(f"Error in split_video_to_frames: {e}")

# Function to perform blob detection
def detect_blobs(image, min_area=1000, max_area=3000, min_circularity=0.1, min_convexity=0.5, min_inertia_ratio=0.01):
    try:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply thresholding
        _, thresh_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY_INV)

        # Set up the blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = min_circularity
        params.filterByConvexity = True
        params.minConvexity = min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = min_inertia_ratio
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(thresh_image)
        logging.info(f"Detected {len(keypoints)} blobs in the image")
        return keypoints
    except Exception as e:
        logging.error(f"Error in detect_blobs: {e}")
        return []

# Function to save blobs as images
def save_blobs_as_images(frame_path, keypoints, output_dir):
    try:
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
                logging.info(f"Saved blob {i} to {blob_path}")
    except Exception as e:
        logging.error(f"Error in save_blobs_as_images: {e}")

# Function to automatically label blobs
def label_blobs_automatically(blob_dir, label_output_file):
    try:
        labels = []
        for file_name in os.listdir(blob_dir):
            if file_name.endswith(".jpg"):
                blob_path = os.path.join(blob_dir, file_name)
                blob = cv2.imread(blob_path)
                height, width, _ = blob.shape

                # Detect keypoints in the blob image itself to check detection
                keypoints = detect_blobs(blob)
                if not keypoints:
                    logging.warning(f"No keypoints detected in {blob_path}")

                for kp in keypoints:
                    center_x = kp.pt[0]
                    
                    # Simple criteria for labeling
                    if center_x < width / 3:
                        label = "Left"
                    elif center_x > 2 * width / 3:
                        label = "Right"
                    else:
                        label = "Center"

                    labels.append((file_name, label))
                    logging.info(f"Labeled blob in {file_name} as {label}")

        with open(label_output_file, 'w') as f:
            for file_name, label in labels:
                f.write(f"{file_name},{label}\n")
        logging.info(f"Labels saved to {label_output_file}")
    except Exception as e:
        logging.error(f"Error in label_blobs_automatically: {e}")

def main():
    # Paths
    video_path_1 = "/Users/ayushpuri/Desktop/DRC TEAM 1/video4.avi"
    video_path_2 = "/Users/ayushpuri/Desktop/DRC TEAM 1/video5.avi"
    frames_output_dir = "/Users/ayushpuri/Desktop/DRC TEAM 1/Dataset/Frames"
    blobs_output_dir = "/Users/ayushpuri/Desktop/DRC TEAM 1/Dataset/Blobs"

    # Create output directories if they don't exist
    os.makedirs(frames_output_dir, exist_ok=True)
    os.makedirs(blobs_output_dir, exist_ok=True)

    # Split videos into frames
    split_video_to_frames(video_path_1, frames_output_dir)
    split_video_to_frames(video_path_2, frames_output_dir)

    # Process each frame for blob detection and save blobs
    def process_frame(frame_file):
        frame_path = os.path.join(frames_output_dir, frame_file)
        frame_image = cv2.imread(frame_path)
        keypoints = detect_blobs(frame_image)
        save_blobs_as_images(frame_path, keypoints, blobs_output_dir)

    with ThreadPoolExecutor() as executor:
        executor.map(process_frame, os.listdir(frames_output_dir))

    # Automatically label blobs
    label_output_file = '/Users/ayushpuri/Desktop/DRC TEAM 1/Dataset/blob_labels.csv'
    label_blobs_automatically(blobs_output_dir, label_output_file)

    logging.info("Processing completed.")

if __name__ == "__main__":
    main()