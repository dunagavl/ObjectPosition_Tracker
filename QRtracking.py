import cv2

camera_id = 0
delay = 1
window_name = 'OpenCV QR Code'

qcd = cv2.QRCodeDetector()
cap = cv2.VideoCapture(camera_id)

while True:
    ret, frame = cap.read()

    if ret:
        ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
        if ret_qr:
            for s, p in zip(decoded_info, points):
                if s:
                    color = (0, 255, 0)
                    # Calculate centroid coordinates
                    centroid_x = int(p[:, 0].mean())
                    centroid_y = int(p[:, 1].mean())
                    # Print decoded text + coordinates
                    print(f"QR Code: {s}")
                    print(f"Corners: {p.astype(int)}")
                    print(f"Centroid: ({centroid_x}, {centroid_y})\n")
                else:
                    color = (0, 0, 255)
                frame = cv2.polylines(frame, [p.astype(int)], True, color, 8)
                # Optional: Draw centroid
                if s:
                    cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
        cv2.imshow(window_name, frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)
cap.release()