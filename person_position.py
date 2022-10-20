import cv2
import numpy as np



class PersonPosition:

    def __init__(self):
        a = 1
    

    def facial_recognition(self, frame, angleRange, colorCircle, face_mesh1):

        angles = []
        pNb = -1

        for i in range(0, 360, angleRange):

            # Search a result of facial recognition in different box
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

                        # Calculate the angle
                        if(i < 180):
                            angles.append(int(cordX*180/1920))
                        else:
                            angles.append(int(180 + cordX*180/1920))

                        # Show on the video a bow where are the persons
                        cv2.circle(frame, (cordX, cordY), 4, colorCircle[pNb], -1)
                        cv2.rectangle(frame, (cordX-40, cordY-40), (cordX+40, cordY+40), colorCircle[pNb], 2)

                        # Show on the angle scale where are the persons
                        if(i < 180):
                            cv2.circle(frame, (cordX, 15), 6, (0, 0, 255), -1)
                        else:
                            cv2.circle(frame, (cordX, 550), 6, (0, 0, 255), -1)
        
        return angles