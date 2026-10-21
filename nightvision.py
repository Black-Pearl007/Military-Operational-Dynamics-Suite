import cv2
import numpy as np


def apply_night_vision_effect(frame, gamma=0.6):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Denoise first (CLAHE would otherwise amplify sensor noise)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. Local contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    boosted = clahe.apply(denoised)

    # 3. Gamma < 1 brightens dark regions without blowing out bright ones
    lut = np.array([255 * (i / 255) ** gamma for i in range(256)], dtype=np.uint8)
    brightened = cv2.LUT(boosted, lut)

    # 4. Unsharp mask (OpenCV only, no scikit-image needed)
    blur = cv2.GaussianBlur(brightened, (0, 0), 2)
    sharpened = cv2.addWeighted(brightened, 1.5, blur, -0.5, 0)

    # 5. Green night-vision tint
    out = np.zeros_like(frame)
    out[:, :, 1] = sharpened
    return out


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        nv = apply_night_vision_effect(frame)
        cv2.imshow("Normal vs Enhanced Night Vision", np.hstack((frame, nv)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
