import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe
import pathlib

# Grab cascade file from cv2 for face detection
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

# create classifier to detect objects
clf = cv2.CascadeClassifier(str(cascade_path))

cap = cv2.VideoCapture(0)

# initialize face mesh detector for distance, only show 1 face
detector = FaceMeshDetector(maxFaces=1)

# distance detector function, inputs video frame image
def distance_detector(img):

    # return image and face detection from FaceMesh function
    img, faces = detector.findFaceMesh(img, draw=False)

    # if face detected, include in faces list.
    # Draw circle where left eye is detected with radius of 5.
    # Draw circle where right eye is detected with radius of 5.
    # Draw line inbetween circles with distance of left eye and right eye
    if faces:
        face = faces[0]
        pointLeft = face[145]  # left eye
        pointRight = face[374]  # right eye

        # optional draw on face:
        # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 1)
        # cv2.circle(img, pointLeft, 1, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, pointRight, 1, (255, 0, 255), cv2.FILLED)

        # find distance of line in pixels (width of pixels w)
        w, _ = detector.findDistance(pointLeft, pointRight)

        # finding focal length
        W = 6.3  # average width between eyes in cm
        d = 53  # distance between user and camera in cm
        f = (w * d) / W  # focal length formula to find average focal length

        # Finding distance (or depth) from camera to user with focal length average
        f = 555
        d = (W * f) / w  # in cm

        # convert cm to inches and display on image
        inches = d * 0.39
        print(f'Current Distance: {inches:.1f} inches')
        cvzone.putTextRect(img, f'Distance: {inches:.1f} in', (face[10][0] - 100, face[10][1] - 50), scale=1.5)

    return img

# Face Rectangle function (I MADE)
def face_rectangle(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # draw rectangle around face on frame, pink with thickness of 2
    for (x, y, width, height) in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 210, 0), 2)

    return image

while True:

    success, img = cap.read()

    # distance function and face detection function
    image = distance_detector(img)
    face_rectangle(image)

    cv2.imshow("image", image)

    if cv2.waitKey(1) == ord('q'):
        print('\n')
        print("DISTANCE PROGRAM STOPPED")
        break