import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import math

# Parameters
offset = 20
imgSize = 300
folder = "Numbers1"
labels = ['0','1','2','3','4','5','6','7','8','9']
images_per_label = 150

# Create folders for each label
for label in labels:
    label_folder = os.path.join(folder, label)
    os.makedirs(label_folder, exist_ok=True)

# Initialize
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
current_label_index = 0  # Start with the first label
image_counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to access the webcam.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y - offset):min(y + h + offset, img.shape[0]),
                      max(0, x - offset):min(x + w + offset, img.shape[1])]

        try:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Show cropped and resized image
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            # Save image to the current label folder
            current_label = labels[current_label_index]
            if image_counter < images_per_label:
                save_path = os.path.join(folder, current_label, f"{image_counter+150}.jpg")
                cv2.imwrite(save_path, imgWhite)
                image_counter += 1
                print(f"Saved {save_path}")

        except Exception as e:
            print("Error processing hand:", e)

    # Show current label on screen
    label_text = f"Capturing for: {labels[current_label_index]}"
    cv2.putText(imgOutput, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display image with text
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1) & 0xFF

    # Switch label on spacebar press
    if key == 32:  # Spacebar to change label
        current_label_index = (current_label_index + 1) % len(labels)
        image_counter = 0  # Reset counter for new label
        print(f"Switched to label: {labels[current_label_index]}")

    # Exit on ESC key
    elif key == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
