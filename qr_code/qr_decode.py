import cv2
import numpy as np
from pyzbar.pyzbar import decode

def main():
    # 1. Start the webcam
    cap = cv2.VideoCapture(0)

    print("Scanning for QR codes... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Find and Decode QR Codes
        # 'decode' returns a list of all barcodes/QR codes found in the frame
        decoded_objects = decode(frame)

        for obj in decoded_objects:
            # --- Extract Data ---
            # Data is returned as bytes, so we decode to string
            qr_data = obj.data.decode('utf-8')
            qr_type = obj.type  # e.g., "QRCODE", "EAN13"

            # --- Draw Bounding Box ---
            # obj.polygon contains the 4 corner points of the QR code
            points = obj.polygon
            
            # If points are found, draw lines connecting them
            if len(points) == 4:
                pts = np.array(points, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Draw the polygon (green box)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            else:
                # Fallback to simple rectangle if polygon is weird
                x, y, w, h = obj.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # --- Draw Text ---
            # Put the decoded text just above the QR code
            # We use the rect to find a good spot for text
            x, y, w, h = obj.rect
            cv2.putText(
                frame, 
                f"{qr_type}: {qr_data}", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )
            
            # Optional: Print to console
            print(f"Found {qr_type}: {qr_data}")

        # 3. Show the result
        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
