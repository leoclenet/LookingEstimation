import math
import cv2
import mediapipe as mp
import numpy as np
import time as tm
from look_estimator import LookEstimator



def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        look_estimator.change_position(x, y)
    if event == cv2.EVENT_MBUTTONDOWN:
        print("middle")

 

'''
1. Choose between importing a video or using a webcam in live
2. Set the parameters (angles & names of each persons, framePrecision
    & detection/tracking confidence)
3. Execute main.py with your terminal
'''


cap = cv2.VideoCapture('/home/kali/Documents/Stage/Sources/video4.mp4') # Video
# cap = cv2.VideoCapture(0) # Camera

# Number of images per second
frameRate = 5

# Parameters
names = ['0', '1', '2', '3', '4', '5']
angleRange = 15


# Frame size precision
framePrecision = 130

# Detection & tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh1 = [mp_face_mesh.FaceMesh(min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6) for x in names]

face_mesh2 = [mp_face_mesh.FaceMesh(min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6) for x in names]

# Initialize
look_estimator = LookEstimator(names, frameRate)

# get the colors
colorCircle = look_estimator.get_colors()



while cap.isOpened():

    success, frame = cap.read()

    angles = []

    # Draw graduation angle on video
    look_estimator.graduation_angle_video(frame)

    pNb = -1

    for i in range(0, 360, angleRange):

        if(i < 180):
            x1 = int(i*1920/180)
            x2 = int((i+angleRange)*1920/180)
            y1 = 0
            y2 = 540
            cutf = frame[y1:y2, x1:x2]
        else:
            x1 = int(i*1920/360)
            x2 = int((i+angleRange)*1920/360)
            y1 = 540
            y2 = 1080
            cutf = frame[y1:y2, x1:x2]

        cutf = cv2.cvtColor(cutf, cv2.COLOR_BGR2RGB)
        results = face_mesh1[pNb].process(cutf)

        if results.multi_face_landmarks:

            pNb += 1
            
            # Convert the color space from RGB to BGR to display well with Opencv
            cutf = cv2.cvtColor(cutf, cv2.COLOR_RGB2BGR)

            face_coordination_in_real_world = np.array([
                [285, 528, 200],
                [285, 371, 152],
                [197, 574, 128],
                [173, 425, 108],
                [360, 574, 128],
                [391, 425, 108]
            ], dtype=np.float64)

            h, w, _ = cutf.shape
            face_coordination_in_image = []

            if results.multi_face_landmarks:

                for face_landmarks in results.multi_face_landmarks:


                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [1, 9, 57, 130, 287, 359]:
                            x, y = int(lm.x * w), int(lm.y * h)
                            face_coordination_in_image.append([x, y])

                    face_coordination_in_image = np.array(face_coordination_in_image,
                                                        dtype=np.float64)
                    
                    # Calculate the coord of the persons
                    cordX = int(x1 + face_coordination_in_image[0][0])
                    
                    if(i < 180):
                        cordY = int(face_coordination_in_image[0][1])
                    else:
                        cordY = int(540 + face_coordination_in_image[0][1])
                    
                    cv2.circle(frame, (cordX, cordY), 4, colorCircle[pNb], -1)
                    cv2.rectangle(frame, (cordX-40, cordY-40), (cordX+40, cordY+40), colorCircle[pNb], 2)

                    if(i < 180):
                        cv2.circle(frame, (cordX, 15), 6, (0, 0, 255), -1)
                    else:
                        cv2.circle(frame, (cordX, 550), 6, (0, 0, 255), -1)

                    # Calculate the angle
                    if(i < 180):
                        angles.append(int(cordX*180/1920))
                    else:
                        angles.append(int(180 + cordX*180/1920))


    # Number of persons for each frame
    numberPerson = len(angles)

    # Calculate where are the persons in the frame
    personsFrame = look_estimator.persons_position_in_frame(angles, numberPerson)

    # Write the names of persons in frame
    look_estimator.draw_persons_in_frame(frame, names, framePrecision, numberPerson)

    # Draw cut
    look_estimator.draw_image_cut(frame, angles, framePrecision, numberPerson)

    # Calculate where are the persons in the circle
    look_estimator.persons_position_in_circle(angles, numberPerson)

    # Draw the situation
    look_estimator.draw_situation(frame, names, numberPerson)

    # Draw graduation angle on schema
    look_estimator.graduation_angle_schema(frame)


    for i in range(0, len(angles)):

        # Cut the image for each persons
        if(angles[i] < 180):
            cut = frame[0:540, personsFrame[i, 0]-framePrecision:personsFrame[i, 0]+framePrecision]
        else:
            cut = frame[540:1080, personsFrame[i, 0]-framePrecision:personsFrame[i, 0]+framePrecision]
        
        # Convert the color space from BGR to RGB and get Mediapipe results
        cut = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
        results = face_mesh2[i].process(cut)

        # Convert the color space from RGB to BGR to display well with Opencv
        cut = cv2.cvtColor(cut, cv2.COLOR_RGB2BGR)

        face_coordination_in_real_world = np.array([
            [285, 528, 200],
            [285, 371, 152],
            [197, 574, 128],
            [173, 425, 108],
            [360, 574, 128],
            [391, 425, 108]
        ], dtype=np.float64)

        h, w, _ = cut.shape
        face_coordination_in_image = []

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_coordination_in_image.append([x, y])

                face_coordination_in_image = np.array(face_coordination_in_image,
                                                    dtype=np.float64)
                    
                # The camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                    [0, focal_length, h / 2],
                                    [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Use solvePnP function to get rotation vector
                success, rotation_vec, transition_vec = cv2.solvePnP(
                    face_coordination_in_real_world, face_coordination_in_image,
                    cam_matrix, dist_matrix)

                # Use Rodrigues function to convert rotation vector to matrix
                rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                result = look_estimator.rotation_matrix_to_angles(rotation_matrix)

                # Calculate & show the results of pitch, yaw & roll
                for l, info in enumerate(zip(('pitch', 'yaw'), result)):
                    k, v = info

                    text = f'{k}: {int(v)}'

                    # Save yaw & pitch to estimate person
                    if(k == 'yaw'): yaw = int(v)
                    if(k == 'pitch'): pitch = int(v)
                    
                    # Show results
                    look_estimator.draw_yaw(frame, text, i, l, framePrecision)

                # Draw looking lines
                look_estimator.draw_lines(frame, yaw, angles, names, i)

                # Looking position
                look_estimator.looking_position(frame, angles, yaw, names, framePrecision, i, numberPerson)

              
        
    cv2.imshow('Head Pose Angles', frame)

    cv2.setMouseCallback('Head Pose Angles', click_event)

    #tm.sleep(0.25)

    # Quit if press Escape
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()

