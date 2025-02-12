import cv2
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize camera and detector
camera_id = 0
delay = 1
window_name = 'OpenCV QR Code'
qcd = cv2.QRCodeDetector()
cap = cv2.VideoCapture(camera_id)

# Data structures
qr_history = defaultdict(lambda: {'timestamps': [], 'x_positions': []})
color_map = {}  # Permanent color storage

# Plot setup
plt.ion()
fig, ax = plt.subplots()
ax.set_title("QR Code X-Position vs Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("X Position (pixels)")
cmap = plt.get_cmap('tab10')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
    current_qrs = set()

    if ret_qr:
        for s, p in zip(decoded_info, points):
            if s:
                current_qrs.add(s)
                centroid_x = int(p[:, 0].mean())
                qr_history[s]['timestamps'].append(current_time)
                qr_history[s]['x_positions'].append(centroid_x)
                frame = cv2.polylines(frame, [p.astype(int)], True, (0, 255, 0), 8)
                cv2.putText(frame, f"{s} ({centroid_x})", tuple(p[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Update plot every 5 frames
    if cv2.getTickCount() % 5 == 0:
        ax.clear()
        ax.set_title("QR Code X-Position vs Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("X Position (pixels)")
        
        # Redraw all lines with current data
        for qr_code, data in qr_history.items():
            if qr_code not in color_map:
                color_map[qr_code] = cmap(len(color_map) % 10)
            
            # Plot entire history
            line = ax.plot(
                data['timestamps'],
                data['x_positions'],
                color=color_map[qr_code],
                label=qr_code,
                marker='o' if qr_code in current_qrs else ''
            )
        
        # Adjust plot limits
        if qr_history:
            all_times = [t for data in qr_history.values() for t in data['timestamps']]
            all_xpos = [x for data in qr_history.values() for x in data['x_positions']]
            ax.set_xlim(min(all_times) - 1, max(all_times) + 1)
            ax.set_ylim(min(all_xpos) - 50, max(all_xpos) + 50)
        
        ax.legend(loc='upper left')
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Display camera feed
    cv2.imshow(window_name, frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Export data
for qr_code, data in qr_history.items():
    print(f"\nData for {qr_code}:")
    print("Time (s) | X Position")
    for t, x in zip(data['timestamps'], data['x_positions']):
        print(f"{t:.2f} | {x}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.close()