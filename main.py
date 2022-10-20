import math
import cv2
import mediapipe as mp
import numpy as np
import time as tm

from person_position import PersonPosition
from person_looking import PersonLooking



def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        person_looking.change_position(x, y)



# Video parameters
cap = cv2.VideoCapture('/home/kali/Documents/Stage/Sources/video4.mp4') # Video
# cap = cv2.VideoCapture(0) # Camera
frameRate = 5   # Number of images per second


# Persons & precision parameters
names = ['0', '1', '2', '3', '4', '5']
angleRange = 15
framePrecision = 130    # Frame size precision


# Detection & tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh1 = [mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5) for x in names]

face_mesh2 = [mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5) for x in names]


# Initialize
person_position = PersonPosition()
person_looking = PersonLooking(names, frameRate)

# Get the colors
colorCircle = person_looking.get_colors()



while cap.isOpened():

    success, frame = cap.read()

    # Determine where are the persons in the frame
    angles = person_position.facial_recognition(frame, angleRange, colorCircle, face_mesh1)
    nbPerson = len(angles)

    # Calculate where are the persons
    personsFrame = person_looking.persons_position_in_frame(angles, nbPerson)

    # Draw the situation on video & schema
    person_looking.draw_on_video(frame, names, framePrecision, nbPerson, angles)
    person_looking.draw_schema(frame, names, nbPerson, angles)


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

                result = person_looking.rotation_matrix_to_angles(rotation_matrix)

                # Calculate & show the results of pitch, yaw & roll
                for l, info in enumerate(zip(('pitch', 'yaw'), result)):
                    k, v = info

                    text = f'{k}: {int(v)}'

                    # Save yaw & pitch to estimate person
                    if(k == 'yaw'): yaw = int(v)
                    if(k == 'pitch'): pitch = int(v)
                    
                # Draw results
                person_looking.draw_result(frame, yaw, pitch, i, framePrecision, angles, names)

                # Looking position
                person_looking.looking_position(frame, angles, yaw, names, framePrecision, i, nbPerson)

              
        
    cv2.imshow('Head Pose Angles', frame)

    cv2.setMouseCallback('Head Pose Angles', click_event)

    # Quit if press Escape
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
