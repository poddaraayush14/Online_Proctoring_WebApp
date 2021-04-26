#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import random as rng
import datetime
import dlib
import face_recognition as fr
import csv
from scipy.spatial import distance


cap = cv2.VideoCapture(0)

roll = input("Enter Roll Number : ")


def calculate_eye(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        eye_aspect_ratio = (A+B)/(2.0*C)
        return eye_aspect_ratio


detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initializations
center_left = 85
center_right = 110

start_of_exam = datetime.datetime.now()
time_stamp = datetime.datetime.now()
next = start_of_exam
diff_time = 1500

out_of_frame = []
motion_stamps = []

max_frame_width = 200
max_frame_height = 120

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7


####Face Recognition


image_directory = 'C:\\Users\\POWER\\Downloads\\WSD Lab Term Project\\Images Directory\\' + str(roll) + '.jpg'
person_image = fr.load_image_file(image_directory)

#person_image = fr.load_image_file(r'C:\Users\POWER\Downloads\WSD Lab Term Project\rishav.jpg')
person_face_encoding = fr.face_encodings(person_image)[0]

known_face_encodings = [person_face_encoding]
known_face_names = ["Rishav"]



####Smartphone and Multiple Person Detector
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))






while True:
    
    ret, frame = cap.read()
    if ret is False:
        break
      
    height,width = frame.shape[:2]   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    ##### Detecting objects
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    count_of_person = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            if label == 'person':
                count_of_person = count_of_person + 1
                
            if count_of_person > 1:
                motion_stamps.append(["Several People", round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)])
                diff = round((datetime.datetime.now()-next).total_seconds()*1000)
                    
                if diff > diff_time :
                    filename = str(round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)) + ".jpg"
                    cv2.imwrite(filename,frame)
                    next = datetime.datetime.now()
                cv2.putText(frame,  
                        'Several People',  
                        (200, 100),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4) 
                
            if label == 'cell phone' :
                diff = round((datetime.datetime.now()-next).total_seconds()*1000)
                    
                if diff > diff_time :
                    filename = str(round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)) + ".jpg"
                    cv2.imwrite(filename,frame)
                    next = datetime.datetime.now()
                motion_stamps.append(["Smartphone", round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)])
                cv2.putText(frame,  
                        'Smart Phone',  
                        (50, 100),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4)
     
    if count_of_person == 0:
        diff = round((datetime.datetime.now()-next).total_seconds()*1000)

        if diff > diff_time :
            filename = str(round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)) + ".jpg"
            cv2.imwrite(filename,frame)
            next = datetime.datetime.now()
        motion_stamps.append(["Person not in frame", round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)])

    
        
    ##### Face Recognition

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom ,left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name == "Unknown":
            cv2.putText(frame,"Not recognised",(200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0), 2, cv2.LINE_4)
            
            if count_of_person == 1 :
                motion_stamps.append(["Unknown Person Detected", round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)])
        
        
        

    ##### Gaze Detector Code - includes eye tracking and head pose detector

    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        sum_x = 0
        sum_y = 0
        
        
        
        ##### Head Pose Estimator
        
        '''tip of nose/left of eye/right of eye/mouth left/mouth right'''
        face_coords = []

        for n in range(32,35):
                sum_x = sum_x + face_landmarks.part(n).x
                sum_y = sum_y + face_landmarks.part(n).y

        #Tip of Nose
        face_coords.append((int(sum_x/3),int(sum_y/3)))

        #Left eye end to tip of nose
        face_coords.append((face_landmarks.part(36).x,face_landmarks.part(36).y))
        ln = distance.euclidean(face_coords[0],face_coords[1])

        #Right eye end to tip of nose
        face_coords.append((face_landmarks.part(45).x,face_landmarks.part(45).y))
        rn = distance.euclidean(face_coords[0],face_coords[2])

        #left mouth end to tip of nose
        face_coords.append((face_landmarks.part(48).x,face_landmarks.part(48).y))
        lmn = distance.euclidean(face_coords[0],face_coords[3])

        #Right mouth end to tip of nose
        face_coords.append((face_landmarks.part(54).x,face_landmarks.part(54).y))
        rmn = distance.euclidean(face_coords[0],face_coords[4])

        gaze_tester = (rn + rmn)/(ln + lmn)
        alignment_tester = (rn + ln)/ (rmn + lmn)
        
        head_down = 0
        head_right = 0
        head_left = 0

        if alignment_tester > 2.2:
            cv2.putText(frame, "Head Down", (150,350), font, 2, (0, 0, 255), 4)
            head_down = 1

            if gaze_tester > 1.18:
                cv2.putText(frame, "Head Right", (50,400), font, 2, (0, 0, 255), 4)
                head_right = 1
            elif gaze_tester < 0.83:
                cv2.putText(frame, "Head Left", (50,400), font, 2, (0, 0, 255), 4)
                head_left = 1
        
        else :
            if gaze_tester > 1.25:
                cv2.putText(frame, "Head Right", (50,400), font, 2, (0, 0, 255), 4)
                head_right = 1
            elif gaze_tester < 0.8:
                cv2.putText(frame, "Head Left", (50,400), font, 2, (0, 0, 255), 4)
                head_left = 1
                
        head_estimator_ud = head_down
        head_estimator_rl = head_right + head_left
                
                
        #### Eye Gaze Estimator
        rightEye = []
        leftEye = []

        for n in range(42,48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x,y))
                next_point = n+1
                if n == 47:
                        next_point = 42
                if n == 42:
                    left_r = x
                if n == 45:
                    right_r = x
                if n == 43:
                    up_r = y
                if n == 47:
                    down_r = y
                    
        for n in range(36,42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x,y))
                next_point = n+1
                if n == 41:
                        next_point = 36
                if n == 36:
                    left_l = x
                if n == 39:
                    right_l = x
                if n == 37:
                    up_l = y
                if n == 41:
                    down_l = y

        right_eye = calculate_eye(rightEye)
        left_eye = calculate_eye(leftEye)

        EYE = (right_eye+left_eye)/2
        EYE = round(EYE,2)

        frame_of_eye_r = frame[up_r-2:down_r+2,left_r-2:right_r+2]
        roi_r = cv2.resize(frame_of_eye_r, (max_frame_width,max_frame_height))

        frame_of_eye_l = frame[up_l-2:down_l+2,left_l-2:right_l+2]
        roi_l = cv2.resize(frame_of_eye_l, (max_frame_width,max_frame_height))

        if EYE > 0.18 :


            # Extracting the height and width of an image 
            rows, cols = roi_r.shape[:2]

            # generating vignette mask using Gaussian 
            # resultant_kernels
            X_resultant_kernel = cv2.getGaussianKernel(cols,200)
            Y_resultant_kernel = cv2.getGaussianKernel(rows,200)
            

            #generating resultant_kernel matrix 
            resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

            #creating mask and normalising by using np.linalg
            # function
            mask = 200 * resultant_kernel / np.linalg.norm(resultant_kernel)


            # applying the mask to each channel in the input image
            for i in range(3):
                roi_r[:,:,i] = roi_r[:,:,i] * mask
                roi_l[:,:,i] = roi_l[:,:,i] * mask




            gray_roi_r = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
            (minVal_r, maxVal_r, minLoc_r, maxLoc_r) = cv2.minMaxLoc(gray_roi_r)
            cv2.circle(roi_r, maxLoc_r, 15, (255, 0, 0), 2)
            

            gray_roi_l = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY)
            (minVal_l, maxVal_l, minLoc_l, maxLoc_l) = cv2.minMaxLoc(gray_roi_l)
            cv2.circle(roi_l, maxLoc_l, 15, (255, 0, 0), 2)
            
            x=maxLoc_r[0]
            a=maxLoc_l[0]
            y=maxLoc_r[1]
            b=maxLoc_l[1]

            if (x<center_left and a<center_left) or (x<center_left and a>center_right) or (a<center_left and x>center_right) or (x>center_right and a>center_right):
                if head_estimator_rl > 0 :
                    motion_stamps.append(["Either Head or Eye Twisted",round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)])
                    cv2.putText(frame,  
                        'Either Head or Eye Twisted ',  
                        (50, 50),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4)
                    diff = round((datetime.datetime.now()-next).total_seconds()*1000)
                    
                    if diff > diff_time :
                        filename = str(round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)) + ".jpg"
                        cv2.imwrite(filename,frame)
                        next = datetime.datetime.now()
                else :
                    
                    cv2.putText(frame,  
                        'Eyeball at Center',  
                        (50, 50),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4)


            else :
                motion_stamps.append(["Gazing",round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)])
                diff = round((datetime.datetime.now()-next).total_seconds()*1000)
                
                if diff > diff_time :
                    filename = str(round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)) + ".jpg"
                    cv2.imwrite(filename,frame)
                    next = datetime.datetime.now()
                cv2.putText(frame,  
                    'Gazing',  
                    (50, 50),  
                    font, 1,  
                    (0, 255, 255),  
                    2,  
                    cv2.LINE_4)


            cv2.imshow("frame",frame)
            

        else:
            if head_estimator_rl > 0 :
                motion_stamps.append(["Head Twisted and Eye closed for Webcam",round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)])
                cv2.putText(frame,  
                    'Head Twisted ',  
                    (50, 50),  
                    font, 1,  
                    (0, 255, 255),  
                    2,  
                    cv2.LINE_4)
                diff = round((datetime.datetime.now()-next).total_seconds()*1000)
                
                if diff > diff_time :
                    filename = str(round((datetime.datetime.now()-start_of_exam).total_seconds()*1000)) + ".jpg"
                    cv2.imwrite(filename,frame)
                    next = datetime.datetime.now()
                
            cv2.putText(frame,  
                        'Closed for webCam',  
                        (300, 100),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4)
            cv2.imshow("frame",frame)



        
    if cv2.waitKey(100) & 0xFF == ord('q'): 

        end_of_exam = datetime.datetime.now()
        exam_duration = round((end_of_exam - start_of_exam).total_seconds()*1000)
        
        #print(motion_stamps)
        print("Exam Duration - ",exam_duration)
        with open('Gazing_instances.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(motion_stamps)
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




