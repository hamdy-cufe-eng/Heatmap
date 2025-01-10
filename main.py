# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Adjust the model path as needed

# Define the path to the match video
video_path = 'testvideo2.mp4'  # Change this to your video path
video = cv2.VideoCapture(video_path)


# Check if the video was loaded successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first few frames for manual player selection
frames = []
for _ in range(6):
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read the video.")
        video.release()
        exit()
    frames.append(frame)

# Initialize trackers and select ROI for each player
trackers = []
for i, frame in enumerate(frames):
    bbox = cv2.selectROI(f"Select Player {i + 1}", frame, False)
    cv2.destroyWindow(f"Select Player {i + 1}")
    tracker = cv2.TrackerCSRT_create()
    try:
        tracker.init(frame, bbox)
        trackers.append(tracker)
    except:
        break

# Define parameters for heatmap generation
heatmap_radius = 2  # Adjust for the radius of influence
field_width = 600  # meters
field_height = 400  # meters

# Initialize separate heatmaps for each player
heatmaps = [np.zeros((field_height, field_width), dtype=np.float32) for _ in range(len(trackers))]

# Load and resize the pitch image to match the heatmap dimensions
pitch_image = cv2.imread('pitch.png')  # Replace with the path to your pitch image
pitch_image_resized = cv2.resize(pitch_image, (field_width, field_height))

# Process each frame in the video
while True:
    ret, frame = video.read()
    if not ret:
        break  # Exit the loop if no frames are left

    for index, tracker in enumerate(trackers):
        try:
            success, bbox = tracker.update(frame)
            if success:
                # Get bounding box coordinates
                x1, y1, w, h = map(int, bbox)
                x_center = int(x1 + w / 2)
                y_center = int(y1 + h / 2)

                # Draw bounding box around the specific player in the video frame
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Tracked Player {index + 1}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Normalize player position to field coordinates
                norm_x = int((x_center / frame.shape[1]) * field_width)
                norm_y = int((y_center / frame.shape[0]) * field_height)


                # Create a circular influence on the heatmap for this player
                for i in range(-heatmap_radius, heatmap_radius + 1):
                    for j in range(-heatmap_radius, heatmap_radius + 1):
                        if i ** 2 + j ** 2 <= heatmap_radius ** 2:  # Check if within the circle
                            x_pos = norm_x + i
                            y_pos = norm_y + j
                            # Accumulate heat only if within bounds of the field dimensions
                            if 0 <= x_pos < field_width and 0 <= y_pos < field_height:
                                heatmaps[index][y_pos, x_pos] += 1.0  # Increment the heatmap at this position
        except Exception as e:
            print(f"Tracking error for player {index + 1}: {e}")
            continue

    # Display the video frame with the tracked players
    cv2.imshow("Video with Player Tracking", frame)

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close display window
video.release()
cv2.destroyAllWindows()
print("Video processing complete.")

# Apply Gaussian filter for smoother heatmaps
heatmaps_smoothed = [gaussian_filter(heatmap, sigma=5) for heatmap in heatmaps]

# Process each heatmap and overlay it on the pitch image
for index, heatmap in enumerate(heatmaps_smoothed):
    # Normalize and apply a color map to the heatmap
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Blend the heatmap with the pitch image
    overlay = cv2.addWeighted(pitch_image_resized, 0.5, heatmap_color, 0.5, 0)

    # Resize the overlay to 400x400 for display
    overlay_resized = cv2.resize(overlay, (400, 400))

    # Display the resized overlay
    cv2.imshow(f"Player {index + 1} Movement Heatmap on Pitch", overlay_resized)
    cv2.waitKey(0)  # Press any key to move to the next heatmap
  # Press any key to move to the next heatmap

# Close the display windows
cv2.destroyAllWindows()
