import cv2, dlib, sys
import cv2 as cv
import numpy as np


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


# face detector
detector = dlib.get_frontal_face_detector()
# 68 points predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ryan = cv2.imread("muhan2.png", cv2.IMREAD_UNCHANGED)

# distinguish points in each area of the face
all = list(range(0, 68))
jawline = list(range(0, 17))
right_eyebrow = list(range(17, 22))
left_eyebrow = list(range(22, 27))
nose = list(range(27, 36))
right_eye = list(range(36, 42))
left_eye = list(range(42, 48))
mouth_outline = list(range(48, 61))
mouth_inline = list(range(61, 68))

index = all

result = None

while True:

    # read the frame
    ret, frame = cap.read()

    if (ret):
        dst = cv2.flip(frame, 1)

        # convert to RGB
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        # detect faces in the frame
        detect = detector(gray, 1)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for face in detect:

            # find 68 points on the face
            shape = predictor(dst, face)

            lists = []
            for p in shape.parts():
                lists.append([p.x, p.y])

            lists = np.array(lists)

            top_left = np.min(lists, axis=0)
            bottom_right = np.max(lists, axis=0)

            center_x, center_y = np.mean(lists, axis=0).astype(np.int)

            for s in lists:
                cv2.circle(frame, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.circle(frame, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(frame, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2,
                       lineType=cv2.LINE_AA)

            cv2.circle(frame, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2,
                       lineType=cv2.LINE_AA)
            for i, point in enumerate(lists[index]):
                points = (point[0], point[1])
                cv.circle(frame, points, 2, (0, 255, 0), -1)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            face_size = max(bottom_right - top_left)
            result = overlay_transparent(dst, ryan, center_x, center_y, overlay_size=(face_size, face_size))
        # show the frame
        cv2.imshow('result', result)

        # if the 'q' key was pressed, break from the loop
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()