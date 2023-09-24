import cv2
import mediapipe as mp
import math as m
import time
from plyer import notification 

# initialises the camera. (First Camera in System)
cap = cv2.VideoCapture(0)

# font type
font = cv2.FONT_HERSHEY_DUPLEX

# colours
blue = (255, 127, 0)
red = (50, 100, 255)
green = (127, 255, 20)
dark_blue = (127, 20, 100)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)


# calculates the distance between two points
def find_distance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


# this function calculates the angle between three points, where the third point is the origin (0,0).
# So it only needs two points as input
def find_angle(x1, y1, x2, y2):
    # here we use the arccosine
    theta = m.acos((y2 - y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi)*theta
    return degree

# A function to send alerts 
def send_alerts(x):
    note = notification.notify(title="WARNING", message="You are in a bad posture", timeout=10)
    return note



# output = pose-estimation via mediapipe
def pose_output():
    # initialises the Pose-Estimation-Pipeline
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # initialise frame counters for good and bad posture time
    good_frames = 0
    bad_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("There is a problem")
            break

        # get fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        # get height and width
        h, w = frame.shape[:2]

        # transforming the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # performing of the pose-estimation on the frame
        results = pose.process(frame_rgb)

        # results.pose_landmarks = Position of the body nodes
        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            lm_pose = mp_pose.PoseLandmark

            # acquiring the coordinates of the needed points
            # Left shoulder
            l_shoulder_x = int(landmarks.landmark[lm_pose.LEFT_SHOULDER].x * w)
            l_shoulder_y = int(landmarks.landmark[lm_pose.LEFT_SHOULDER].y * h)
            # Right shoulder
            r_shoulder_x = int(landmarks.landmark[lm_pose.RIGHT_SHOULDER].x * w)
            r_shoulder_y = int(landmarks.landmark[lm_pose.RIGHT_SHOULDER].y * h)
            # Left ear
            l_ear_x = int(landmarks.landmark[lm_pose.LEFT_EAR].x * w)
            l_ear_y = int(landmarks.landmark[lm_pose.LEFT_EAR].y * h)
            # Left hip
            l_hip_x = int(landmarks.landmark[lm_pose.LEFT_HIP].x * w)
            l_hip_y = int(landmarks.landmark[lm_pose.LEFT_HIP].y * h)

            # calculate distance between left shoulder and right shoulder points
            offset = find_distance(l_shoulder_x, l_shoulder_y, r_shoulder_x, r_shoulder_y)

            # assist to align the camera to point at the side view of the person
            if offset < 100:
                cv2.putText(frame, str(int(offset)) + ' Aligned', (w - 175, 30), font, 0.9, green, 1)
            else:
                cv2.putText(frame, str(int(offset)) + ' Not Aligned', (w - 250, 30), font, 0.9, red, 2)

            # calculate angles of neck and torso inclination
            neck_inclination = find_angle(l_shoulder_x, l_shoulder_y, l_ear_x, l_ear_y)
            torso_inclination = find_angle(l_hip_x, l_hip_y, l_shoulder_x, l_shoulder_y)

            # draw landmarks of shoulder and ear on output
            cv2.circle(frame, (l_shoulder_x, l_shoulder_y), 7, yellow, -1)
            cv2.circle(frame, (l_ear_x, l_ear_y), 7, yellow, -1)

            cv2.circle(frame, (l_shoulder_x, l_shoulder_y - 100), 7, yellow, -1)
            cv2.circle(frame, (r_shoulder_x, r_shoulder_y), 7, pink, -1)
            cv2.circle(frame, (l_hip_x, l_hip_y), 7, yellow, -1)

            cv2.circle(frame, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

            neck_angle_text_string = f"Neck :{str(int(neck_inclination))}"
            torso_angle_text_string = f"Torso :{str(int(torso_inclination))}"

            if neck_inclination < 40 and torso_inclination < 10:
                bad_frames = 0
                good_frames += 1

                cv2.putText(frame, neck_angle_text_string, (10, 30), font, 0.8, light_green, 1)
                cv2.putText(frame, torso_angle_text_string, (10, 60), font, 0.8, light_green, 1)
                cv2.putText(frame, str(int(neck_inclination)), (l_shoulder_x + 10, l_shoulder_y), font, 0.9, light_green, 2)
                cv2.putText(frame, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

                # connect landmarks
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_ear_x, l_ear_y), green, 4)
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_shoulder_x, l_shoulder_y - 100), green, 4)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_shoulder_x, l_shoulder_y), green, 4)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

            else:
                good_frames = 0
                bad_frames += 1

                cv2.putText(frame, neck_angle_text_string, (10, 30), font, 0.8, red, 2)
                cv2.putText(frame, torso_angle_text_string, (10, 60), font, 0.8, red, 2)
                cv2.putText(frame, str(int(neck_inclination)), (l_shoulder_x + 10, l_shoulder_y), font, 0.9, red, 2)
                cv2.putText(frame, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

                # connect landmarks
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_ear_x, l_ear_y), red, 4)
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_shoulder_x, l_shoulder_y - 100), red, 4)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_shoulder_x, l_shoulder_y), red, 4)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(frame, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(frame, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # If you stay in bad posture for more than 3 minutes (180s) send an alert.
        if bad_time > 20:
            send_alerts(bad_time)

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
