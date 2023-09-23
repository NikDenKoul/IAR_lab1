import cv2
import numpy as np
import math


def img_processing(image, img_index, do_output=True, modification_mode='NONE'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cats_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")
    cats = cats_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=1,
        minSize=(30, 30)
    )

    if do_output:
        print(f"Image №{img_index}: found {len(cats)} Cats!")

    if len(cats) == 0:
        return image

    main_cat = cats[0]
    (x, y, w, h) = main_cat

    if modification_mode == 'RECTANGLE':
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif modification_mode == 'BLUR':
        blurred_img = cv2.GaussianBlur(image, (25, 25), 0)

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (x + int(w/2), y + int(h/2)), int(math.sqrt(h*w)/2), [255, 255, 255], -1)

        image = np.where(mask == np.array([255, 255, 255]), image, blurred_img)
    elif modification_mode == 'NONE':
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image = image[y:y + h, x:x + w]

    if do_output:
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite(f"./output/pic_{img_index}.jpg", image)

    return image


def gif_processing(file_path, modification_mode):
    cap = cv2.VideoCapture(file_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            image = img_processing(frame, 0, False, modification_mode)
            cv2.imshow('Frame', image)
        # Break the loop
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Press Q on keyboard to  exit
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
