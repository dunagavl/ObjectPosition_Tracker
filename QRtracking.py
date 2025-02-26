import cv2
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize video file input
video_path = r"C:\Users\Victor\OneDrive\Little Vic\Junior Year\MED Lab\QRTrial_2 - 1740414308309.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize detector
qcd = cv2.QRCodeDetector()
window_name = 'QR Code Tracker'
delay = 1  # Initial delay value, will be adjusted based on video FPS

# Data structures
qr_history = defaultdict(lambda: {'timestamps': [], 'x_positions': []})
color_map = {}  # Permanent color storage

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = frame_count / fps

print(f"Processing video: {video_path}")
print(f"Duration: {video_duration:.2f}s, FPS: {fps:.1f}, Frames: {frame_count}")

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate timestamp based on frame position
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
    
    # Detect QR codes
    ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
    current_qrs = set()

    if ret_qr:
        for s, p in zip(decoded_info, points):
            if s:
                current_qrs.add(s)
                centroid_x = int(p[:, 0].mean())
                qr_history[s]['timestamps'].append(current_time)
                qr_history[s]['x_positions'].append(centroid_x)
                
                # Draw annotations
                frame = cv2.polylines(frame, [p.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(frame, f"{s} ({centroid_x})", tuple(p[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show processed frame
    cv2.imshow(window_name, frame)
    
    # Adjust playback speed to match original FPS
    delay = int(1000 / fps) if fps > 0 else 1
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Generate final plot after processing
print("\nGenerating analysis plot...")
fig, ax = plt.subplots()
ax.set_title("QR Code X-Position Analysis")
ax.set_xlabel("Time (s)")
ax.set_ylabel("X Position (pixels)")
cmap = plt.get_cmap('tab10')

for qr_code, data in qr_history.items():
    color = cmap(len(color_map) % 10)
    color_map[qr_code] = color
    ax.plot(data['timestamps'], data['x_positions'], 
            color=color, label=qr_code, marker='o', markersize=3)

ax.legend(loc='upper left')
ax.set_xlim(0, video_duration)
plt.tight_layout()
plt.show()

# Save results to file
output_filename = video_path.split('.')[0] + '_analysis.txt'
with open(output_filename, 'w') as f:
    for qr_code, data in qr_history.items():
        f.write(f"\nQR Code: {qr_code}\n")
        f.write("Time(s)\tX Position\n")
        for t, x in zip(data['timestamps'], data['x_positions']):
            f.write(f"{t:.3f}\t{x}\n")

print(f"Analysis saved to {output_filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()