import cv2
import mediapipe as mp


# initialises the camera. (First Camera in System)
cap = cv2.VideoCapture(0)


# output = pose-estimation via mediapipe
def pose_output():
    # initialises the Pose-Estimation-Pipeline
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # transforming the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # performing of the pose-estimation on the frame
        results = pose.process(frame_rgb)

        # results.pose_landmarks = Position of the body nodes
        if results.pose_landmarks:
            # projects the nodes on the camera-output
            mp_drawing = mp.solutions.drawing_utils
            # projects the links between the nodes on the camera-output
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # projects
        cv2.imshow("Pose Estimation", frame)

        # loop ends when clicked on esc or q
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


pose_output()


# function, which evaluates the skeleton and gives it a score
def posture_score():
    pass


# function, which gives alerts to the user, when the posture_score is too low.
def score_alerts():
    pass


# function, which saves the posture-Score + timestamp in a csv_file
def posture_score_csv():
    pass


# function, which displays the csv_file as a dashboard. The Posture-Score of the day will be shown.
def dashboard(file):
    pass
