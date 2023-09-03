import cv2

# initialises the camera. (First Camera in System)
cap = cv2.VideoCapture(0)

while True:
    # captures one frame of the camera
    ret, frame = cap.read()

    if not ret:
        break

    # shows the camera-input
    cv2.imshow("Kamera", frame)

    # end the loop, when "q" is clicked
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# function, which calculates the skeleton of a frame, with a CNN (PoseNet)
def detect_skeleton(frame):
    pass


# function, which shows the camera-input, with the skeleton
def show_skeleton():
    pass


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
