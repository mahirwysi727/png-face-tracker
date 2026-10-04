import cv2
import mediapipe as mp
import time
import os
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "facelandmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)


# downloads googles face tracking model for the first time if not downloaded
if not os.path.exists("MODEL_PATH"):
    print("Downloading Face tracking model")
    urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
    print("Model downloaded")



BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode


options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = MODEL_PATH),

    # video mode is simple for v0
    running_mode = RunningMode.VIDEO,

    #only care about 1 face
    num_faces = 1,

    #gives different expressions (smile frown blink jaw)
    output_face_blendshapes = True
)



#0 means the first camera
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Could not open camera.")
    print("If you're using a virtual/phone camera, try changing:")
    print("cv2.VideoCapture(0)")
    print("to:")
    print("cv2.VideoCapture(1)")
    raise SystemExit


with FaceLandmarker.create_from_options(options) as landmarker:

    start_time = time.monotonic()

    smoothed_smile = 0.0

    while True:

        success, frame = cap.read()

        if not success:
            print("could not read camera frame")
            break

        #openCV uses BGR
        #mediapipe excepts RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = rgb_frame
        )

        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        smile_score = 0.0

        if result.face_blendshapes:

            blendshapes = result.face_blendshapes[0]


            # Turn the 52 blendshapes into:
            #
            # {
            #     "mouthSmileLeft": 0.73,
            #     "mouthSmileRight": 0.69,
            #     "eyeBlinkLeft": 0.02,
            #     ...
            # }
            scores = {
                blendshape.category_name: blendshape.score
                for blendshape in blendshapes
            }

            smile_left = scores.get("mouthSmileLeft", 0.0)
            smile_right = scores.get("mouthSmileRight", 0.0)

            smile_score = (smile_left + smile_right) / 2

            alpha = 0.2

            smoothed_smile = (
                alpha * smile_score
                + (1 - alpha) * smoothed_smile
            )


        #simple threshold for v0 
        if state == "NEUTRAL":
            if smoothed_smile > 0.18:
                state = "SMILING"

        elif state == "SMILING":
            if smoothed_smile < 0.08:
                state = "NEUTRAL"


        #show info on camera window
        cv2.putText(
            frame,
            f"Smile: {smile_score:.2f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            state,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        cv2.imshow("PNGTuber Face Tracker", frame)

        #press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

