from telnetlib import X3PAD
from tkinter.font import names
import cv2
import numpy as np
import math


class LookEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, names, frameRate):

        # Schematic parameters
        self.schemaPos = (960, 540)
        
        self.circle_color = (255, 100, 0)
        self.names_color = (255, 102, 255)
        self.line_color = (1, 1, 1)
        self.graduation_color = (255, 255, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.colorCircle = [(0, 128, 255),
                            (0, 255, 255),
                            (0, 255, 0),
                            (255, 255, 0),
                            (255, 0, 0),
                            (255, 0, 255),
                            (128, 128, 128)]


        self.time = [0 for x in names]
        self.personLooking = [None for x in names]

        self.frameRate = frameRate
    

    def get_colors(self):
        return self.colorCircle


    def change_position(self, x, y):
        self.schemaPos = (x, y)


    def rotation_matrix_to_angles(self, rotation_matrix):
        """
        Source: https://github.com/shenasa-ai/head-pose-estimation
        Calculate Euler angles from rotation matrix.
        :param rotation_matrix: A 3*3 matrix with the following structure
        [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
        [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
        [  -Siny             CosySinx                   Cosy*Cosx         ]
        :return: Angles in degrees for each axis
        """
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                        rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi



    def persons_position_in_circle(self, angles, numberPerson):
        # Radius of circle
        self.r = 175

        # Persons Positions
        self.personsCircle = {}

        # Persons positions in the circle
        for i in range(0, numberPerson):
            x = - int(self.r * math.cos((angles[i] * 3.14) / 180)) + self.schemaPos[0]
            y = - int(self.r * math.sin((angles[i] * 3.14) / 180)) + self.schemaPos[1]
            self.personsCircle[i, 0] = x
            self.personsCircle[i, 1] = y


    
    def persons_position_in_frame(self, angles, numberPerson):
        # Persons Positions
        self.personsFrame = {}

        # Persons positions in the frame
        for i in range(0, numberPerson):
            if(angles[i] < 180):
                x = int(angles[i] * 1920 / 180)
                y = 500
            else:
                x = int((angles[i] - 180) * 1920 / 180)
                y = 950
            self.personsFrame[i, 0] = x
            self.personsFrame[i, 1] = y
        return self.personsFrame



    def draw_situation(self, frame, names, numberPerson):
        # Draw rond table
        cv2.circle(frame, self.schemaPos, self.r, (255, 255, 255), -1)

        # Draw camera
        cv2.circle(frame, self.schemaPos, 5, (1, 1, 1), -1)

        # Persons positions in the circle
        for i in range(0, numberPerson):
            cv2.circle(frame, (self.personsCircle[i, 0], self.personsCircle[i, 1]), 18, self.colorCircle[i], -1)
            cv2.putText(frame, names[i], (self.personsCircle[i, 0] - 5, self.personsCircle[i, 1] + 5), self.font, 0.4, self.names_color, 2, cv2.LINE_4)



    def draw_lines(self, frame, yaw, angles, names, i):
        # length of the line
        length = 100
                
        x2 = int(self.personsCircle[i, 0] + length * math.cos((angles[i] - yaw) * 3.14 / 180.0))
        y2 = int(self.personsCircle[i, 1] + length * math.sin((angles[i] - yaw) * 3.14 / 180.0))

        cv2.line(frame, (self.personsCircle[i, 0], self.personsCircle[i, 1]), (x2, y2), self.line_color, 3)

        # So that the lines gets under the circle
        cv2.circle(frame, (self.personsCircle[i, 0], self.personsCircle[i, 1]), 18, self.colorCircle[i], -1)
        cv2.putText(frame, names[i], (self.personsCircle[i, 0] - 5, self.personsCircle[i, 1] + 5), self.font, 0.4, self.names_color, 2, cv2.LINE_4)



    def draw_persons_in_frame(self, frame, names, framePrecision, numberPerson):
        for i in range(0, numberPerson):
            cv2.putText(frame, names[i], (self.personsFrame[i, 0] - framePrecision + 10, self.personsFrame[i, 1] - 50), self.font, 1, self.names_color, 2, cv2.LINE_4)

    

    def draw_yaw(self, frame, text, i, l, framePrecision):
        cv2.putText(frame, text, (self.personsFrame[i, 0] - framePrecision + 10, l*30 + self.personsFrame[i, 1] - 15), self.font, 0.6, (0, 255, 255), 2)
    


    def draw_image_cut(self, frame, angles, framePrecision, numberPerson):

        for i in range(0, numberPerson):
            if(angles[i] < 180):
                cv2.rectangle(frame, ((self.personsFrame[i, 0] - framePrecision), 15),
                            ((self.personsFrame[i, 0] + framePrecision), 530), (0, 0, 255), 4)
            else:
                cv2.rectangle(frame, ((self.personsFrame[i, 0] - framePrecision), 550),
                            ((self.personsFrame[i, 0] + framePrecision), 1060), (0, 0, 255), 4)



    def graduation_angle_video(self, frame):
        cv2.line(frame, (0, 15), (1920, 15), self.graduation_color, 2)
        cv2.line(frame, (0, 550), (1920, 550), self.graduation_color, 2)

        for i in range(0, 181, 10):
            x1 = int(i*1920/181)
            x2 = int((i-5) *1920/181)

            cv2.line(frame, (x1, 10), (x1, 20), self.graduation_color, 2)
            cv2.line(frame, (x2, 13), (x2, 17), self.graduation_color, 2)

            cv2.line(frame, (x1, 545), (x1, 555), self.graduation_color, 2)
            cv2.line(frame, (x2, 548), (x2, 552), self.graduation_color, 2)

            cv2.putText(frame, str(i), (x1 - 15, 40), self.font, 0.6, self.graduation_color, 2)
            cv2.putText(frame, str(i + 180), (x1 - 15, 575), self.font, 0.6, self.graduation_color, 2)



    def graduation_angle_schema(self, frame):
        for i in range(0, 180, 10):
            x1 = - int(self.r * math.cos((i * 3.14) / 180)) + self.schemaPos[0]
            y1 = - int(self.r * math.sin((i * 3.14) / 180)) + self.schemaPos[1]

            x2 = - int(self.r * math.cos(((i + 180) * 3.14) / 180)) + self.schemaPos[0]
            y2 = - int(self.r * math.sin(((i + 180) * 3.14) / 180)) + self.schemaPos[1]

            cv2.circle(frame, (x1, y1), 3, (0, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 3, (0, 0, 255), -1)

            cv2.putText(frame, str(i), (x1 - 10, y1 - 10), self.font, 0.3, (0, 0, 255), 1)
            cv2.putText(frame, str(i + 180), (x2 - 10, y2 - 10), self.font, 0.3, (0, 0, 255), 1)




    def looking_position(self, frame, angles, yaw, names, framePrecision, i, numberPerson):

        lookingAngle = angles[i] + 180 - 2 * yaw

        if(lookingAngle > 360):
            lookingAngle = lookingAngle - 360

        x = - int(self.r * math.cos((lookingAngle * 3.14) / 180)) + self.schemaPos[0]
        y = - int(self.r * math.sin((lookingAngle * 3.14) / 180)) + self.schemaPos[1]
        
        cv2.circle(frame, (x, y), 10, self.colorCircle[i], -1)

        # This version tells which person you are most possibly looking at
        old_min = 360

        for l in range(0, numberPerson):

            new_min = abs(lookingAngle - angles[l])

            if(old_min > new_min):
                old_min = new_min
                mostPossPerson = l
        
        # Person looking text
        if(-20 < old_min < 20):
            color = (0, 255, 0) # Green: we are almost sure

        elif(-40 < old_min < -20 or 20 < old_min < 40):
            color = (0, 255, 255) # Yellow: it's possible

        else:
            color = (0, 0, 255) # Red: not sure
        
        cv2.putText(frame, names[mostPossPerson], (self.personsFrame[i, 0] + framePrecision - 100, self.personsFrame[i, 1] - 15), self.font, 0.7, color, 2)
        
        # Angle difference from the person
        cv2.putText(frame, str(old_min) + "'", (self.personsFrame[i, 0] + framePrecision - 40, self.personsFrame[i, 1] + 15), self.font, 0.55, (0, 255, 255), 2)


        # Calc time looking
        if(self.personLooking[i] != mostPossPerson):
            #Changement de personne
            self.time[i] = 0
            self.personLooking[i] = mostPossPerson
            
        else:
            self.time[i] += 1
        
        cv2.putText(frame, str(int(self.time[i]/(self.frameRate * 10))) + "s", (self.personsFrame[i, 0] + framePrecision - 100, self.personsFrame[i, 1] + 15), self.font, 0.55, (0, 255, 255), 2)

