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


        # Default values if MediaPipe cannot currently see a face
        smile = 0.0

        blink_left = 0.0
        blink_right = 0.0

        jaw_open = 0.0

        brow_down_left = 0.0
        brow_down_right = 0.0

        brow_inner_up = 0.0
 
        squint_left = 0.0
        squint_right = 0.0

        frown_left = 0.0
        frown_right = 0.0


        if result.face_blendshapes:

            blendshapes = result.face_blendshapes[0]

            # Convert the 52 MediaPipe blendshapes into a dictionary.
            scores = {
                blendshape.category_name: blendshape.score
                for blendshape in blendshapes
            }


            # -------------------------
            # MOUTH
            # -------------------------

            smile_left = scores.get("mouthSmileLeft", 0.0)
            smile_right = scores.get("mouthSmileRight", 0.0)

            smile = (smile_left + smile_right) / 2


            frown_left = scores.get("mouthFrownLeft", 0.0)
            frown_right = scores.get("mouthFrownRight", 0.0)


            jaw_open = scores.get("jawOpen", 0.0)


            # -------------------------
            # EYES
            # -------------------------

            blink_left = scores.get("eyeBlinkLeft", 0.0)
            blink_right = scores.get("eyeBlinkRight", 0.0)

            squint_left = scores.get("eyeSquintLeft", 0.0)
            squint_right = scores.get("eyeSquintRight", 0.0)


            # -------------------------
            # EYEBROWS
            # -----------------------       

            brow_down_left = scores.get("browDownLeft", 0.0)
            brow_down_right = scores.get("browDownRight", 0.0)

            brow_inner_up = scores.get("browInnerUp", 0.0)


            #temp state detection
            if smile > 0.18:
                state = "HAPPY"
            else:
                state = "NEUTRAL"
            

        #show info on camera window
        # Everything we want displayed on screen.
        debug_values = [
            ("Smile", smile),

            ("Blink L", blink_left),
            ("Blink R", blink_right),

            ("Jaw Open", jaw_open),

            ("Brow Down L", brow_down_left), 
            ("Brow Down R", brow_down_right),

            ("Brow Inner Up", brow_inner_up),

            ("Squint L", squint_left),
            ("Squint R", squint_right),

            ("Frown L", frown_left),
            ("Frown R", frown_right),
        ]


        # Starting position
        x = 20
        y = 40

        line_height = 32


        for name, value in debug_values:

            text = f"{name:<14} {value:.2f}"

            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            y += line_height


        # Display current state underneath everything.
        cv2.putText(
            frame,
            f"STATE: {state}",
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

        cv2.imshow("PNGTuber Face Tracker", frame)

        #press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

