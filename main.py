import cv2

# Cats detector

# Images processing
image_path = './pictures/pic_'
for i in range(1, 3):
    image = cv2.imread(f"{image_path}{i}.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    catsCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")
    cats = catsCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=1,
        minSize=(30, 30)
    )

    print(f"Image №{i}: found {len(cats)} Cats!")

    cat = 1
    for (x, y, w, h) in cats:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]

        print("[INFO] Object found. Saving locally.")

        cv2.imwrite(f"./output/img_{i}_cat_{cat}.jpg", roi_color)
        cat += 1
