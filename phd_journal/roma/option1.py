import cv2
import numpy as np
import os

# Parameters
frame_rate = 80  # Hz
period = 1.0  # seconds
total_frames = int(frame_rate * period)
width, height = 1920, 1080
circle_radius = 30
circle_distance = 100  # Distance between the centers of the circles

# Create output directory
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)
"""
# Function to calculate circle position
def get_circle_position(frame_num, total_frames, width):
    half_period_frames = total_frames // 2
    max_x = width - circle_radius - circle_distance
    min_x = circle_radius
    if frame_num < half_period_frames:
        x = int((max_x - min_x) * (frame_num / half_period_frames)) + min_x
    else:
        x = int((max_x - min_x) * ((total_frames - frame_num) / half_period_frames)) + min_x
    return x


# Generate frames and write to video
for frame_num in range(total_frames):
    # Create a white background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Calculate positions
    x = get_circle_position(frame_num, total_frames, width)
    y = height // 2

    # Draw the first circle (10 Hz flashing)
    if (frame_num // 4) % 2 == 0:
        cv2.circle(frame, (x, y), circle_radius, (0, 0, 0), -1)

    # Draw the second circle (40 Hz flashing)
    if frame_num % 2 == 0:
        cv2.circle(frame, (x + circle_distance, y), circle_radius, (0, 0, 0), -1)

print("Frams generated successfully.")
"""


# Set up video writer
video_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'avc1')
video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, (width, height))
DURATION = 15  # minutes

# Get list of frame filenames
frame_filenames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')])

# Insert frames
# number of times to repeat one period
for i in range(int(DURATION*60/period)):
    for filename in frame_filenames:
        # Write the frame to the video
        video_writer.write(filename)

# Release the video writer
video_writer.release()

print(f"Video generated successfully: {video_filename}")
