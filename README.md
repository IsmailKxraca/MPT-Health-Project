# MPT-Health-Project
Hochschule Düsseldorf: Machine Perception and Tracking 

Project: Healthy Coding!(& Desk work)

Posture Detection for Better Ergonomics

# Objective  
Develop a machine perception system that uses computer vision techniques to monitor and evaluate users’ posture during work hours, to promote 
better ergonomics and health.

# How it works 
- First of all it looks if the body parts are aligned ( left and right shoulders (pink landmark)) , so that the camera has a 90 degree angle side view of the User
- It calculates the angle of inclination of the neck and the back from the origin. If the angle of inclination of the neck is more than 40 degrees, then the user has an unhealthy posture according to ergonomic studies.
- If the back has an inclination angle of more than 10 degrees, then the user has an unhealthy posture.


![good](https://github.com/IsmailKxraca/MPT-Health-Project/assets/103109804/2da8956b-3fb1-4eba-a399-2a6cd1c94681)

- It then calculates the duration in which the user has an unhealty posture
- If the user has an unhealthy posture for 20 seconds, the user will recieve an alert saying that he has to change his posture

![bad](https://github.com/IsmailKxraca/MPT-Health-Project/assets/103109804/a8f685cc-5f53-4a23-8f0c-c84cb55e2d6b)


# How to use 
- Clone the Respository in your local computer
```
git clone https://github.com/IsmailKxraca/MPT-Health-Project.git
```

- Use the powershell to launch the programm
```
streamlit run main.py
```

# Install required packages
To install the required libraries 
```
pip install -r requirements.txt
```

