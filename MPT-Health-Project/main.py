import cv2
import os
import mediapipe as mp
import math as m
import time
import pandas as pd
import numpy as np
from plyer import notification
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st


# initialises the camera. (First Camera in System)
cap = cv2.VideoCapture(0)

# font type
font = cv2.FONT_HERSHEY_DUPLEX

# colours
blue = (255, 127, 0)
red = (50, 100, 255)
green = (127, 255, 20)
dark_blue = (204, 0, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

warning = cv2.imread("posture_warning.png")

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
def send_alerts():
    note = notification.notify(title="WARNING", message="You are in a bad posture", timeout=10)
    return note


# data is a list with timestamp and good/bad Posture
def safe_in_csv(name, data):
    with open(f"{str(name)}.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


# creates a csv file for the datetime, when the code was run
def create_csv(time):
    header = ["Time", "Posture"]
    with open(f"{str(time)}.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


# output = pose-estimation via mediapipe
def pose_output():
    # initialises the Pose-Estimation-Pipeline
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # initialise frame counters for good and bad posture time
    good_frames = 0
    bad_frames = 0

    now = datetime.now()

    hour = now.strftime('%H')
    minute = now.strftime('%M')
    second = now.strftime('%S')
    millisecond = now.microsecond // 1000

    start_time = f"{hour}:{minute}:{second}.{millisecond}"
    csv_name = f"{hour}_{minute}_{second}"

    create_csv(csv_name)

    bad_warning = 0

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
            back_inclination = find_angle(l_hip_x, l_hip_y, l_shoulder_x, l_shoulder_y)

            # draw landmark of left shoulder
            cv2.circle(frame, (l_shoulder_x, l_shoulder_y), 7, yellow, -1)
            # draw landmark of left ear
            cv2.circle(frame, (l_ear_x, l_ear_y), 7, yellow, -1)
            # draw neck angel-help point
            cv2.circle(frame, (l_shoulder_x, l_shoulder_y - 100), 7, dark_blue, -1)
            # draw landmark of right shoulder
            cv2.circle(frame, (r_shoulder_x, r_shoulder_y), 7, pink, -1)
            # draw landmark of left hip
            cv2.circle(frame, (l_hip_x, l_hip_y), 7, yellow, -1)
            # draw back-angle-help point
            cv2.circle(frame, (l_hip_x, l_hip_y - 100), 7, dark_blue, -1)

            neck_angle_text_string = f"Neck :{str(int(neck_inclination))}"
            back_angle_text_string = f"Back :{str(int(back_inclination))}"

            frame_time = f"{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}.{datetime.now().microsecond // 1000}"

            if neck_inclination < 40 and back_inclination < 10:
                bad_frames = 0
                good_frames += 1

                cv2.putText(frame, neck_angle_text_string, (10, 30), font, 0.8, light_green, 1)
                cv2.putText(frame, back_angle_text_string, (10, 60), font, 0.8, light_green, 1)
                cv2.putText(frame, str(int(neck_inclination)), (l_shoulder_x + 10, l_shoulder_y), font, 0.9, light_green, 2)
                cv2.putText(frame, str(int(back_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

                # connect landmarks
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_ear_x, l_ear_y), green, 4)
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_shoulder_x, l_shoulder_y - 100), green, 4)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_shoulder_x, l_shoulder_y), green, 4)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

                # save the datetime as good
                safe_in_csv(csv_name, [frame_time, "good"])

            else:
                good_frames = 0
                bad_frames += 1

                cv2.putText(frame, neck_angle_text_string, (10, 30), font, 0.8, red, 2)
                cv2.putText(frame, back_angle_text_string, (10, 60), font, 0.8, red, 2)
                cv2.putText(frame, str(int(neck_inclination)), (l_shoulder_x + 10, l_shoulder_y), font, 0.9, red, 2)
                cv2.putText(frame, str(int(back_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

                # connect landmarks
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_ear_x, l_ear_y), red, 4)
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_shoulder_x, l_shoulder_y - 100), red, 4)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_shoulder_x, l_shoulder_y), red, 4)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

                # save the datetime as bad
                safe_in_csv(csv_name, [frame_time, "bad"])

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames


        # pose time
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(frame, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(frame, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # If you stay in bad posture for more than 20 seconds send an alert
        if bad_time > 20:
            if bad_warning == 0:
                send_alerts()
                bad_warning = 1
        else:
            bad_warning = 0

        # projects
        cv2.imshow("Pose Estimation", frame)

        # loop ends when clicked on esc or q
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# deletes all csv-files in the folder, which are older than 24h
def delete_old_csv_files():

    current_time = datetime.now()
    files = os.listdir()
    # all csv-files
    csv_files = [file for file in files if file.endswith(".csv")]

    for file in csv_files:
        file_path = os.path.join(os.getcwd(), file)
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

        # calculate the time-difference
        time_difference = current_time - file_time

        # if the difference is more than 24 hours, delete the file
        if time_difference.total_seconds() > 24 * 3600:
            os.remove(file_path)


# function, which displays the csv_file as a dashboard. The Posture-Score of the day will be shown.
def dashboard(file_name):
    df = pd.read_csv(file_name, delimiter=",")
    df['Posture'] = df['Posture'].replace({'good': 1, 'bad': 2})

    # create line-plot
    plt.figure(figsize=(10, 6))

    x = df['Time']
    y = df['Posture']

    # change y-axis, that 1 is good and 2 is bad
    plt.yticks([1, 2], ['good', 'bad'])

    border = 1.5
    colors = []
    # fills colors with red, if the value is over 1.5 (bad posture) and green if its under 1.5 (good posture)
    for yi in y:
        if yi > border:
            colors.append('red')
        else:
            colors.append('green')

    # plot the line-plot and scatter plot
    plt.plot(x, y, marker='o', linestyle='-')
    plt.scatter(x, y, marker='o', color=colors, zorder=2)

    # axis-labels
    plt.xlabel('Time')
    plt.ylabel('Posture')

    # show plot
    plt.title(file_name[:-4])
    plt.grid(True)
    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def show_24h_dashboard():
    files = os.listdir()
    # all csv-files
    csv_files = [file for file in files if file.endswith(".csv")]

    for file in csv_files:
        df = pd.read_csv(file, delimiter=",")
        st.subheader(f"Report of {file[0:2]}:{file[3:5]} o'clock")

        dashboard(file)


def main():
    """MPT-Health_project"""

    menu = ["Start", "Report"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Start":
        st.title("Detecting Good and Bad Postures")
        st.subheader(" About our Project")
        st.write(
            " Our Project consist of detecting if someone has a bad posture in real-time using advanced methods and "
            "sending alerts to the person saying he has to change his position")
        st.write("This is mainly done by calculating the the angle of inclination of the neck and the back. "
                 "If the neck is inclined at an angle of 40 degrees and the back at 10 degrees, then the person has a "
                 "bad position according to ergonomics rules")
        if st.button("Click to Start"):
            pose_output()

    if choice == "Report":
        delete_old_csv_files()
        st.title("Report of the last 24 Hours")
        show_24h_dashboard()


main()
#pose_output()
#dashboard()
#show_24h_dashboard()