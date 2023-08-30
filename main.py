import cv2 as cv;
from cvzone.FaceDetectionModule import FaceDetector
import pickle;
import face_recognition;
import numpy as np;
import cvzone
import os


cap = cv.VideoCapture(0)
cap.set(3, 640) #width
cap.set(4, 480) #height
imgBackground = cv.imread('resources/backgroundFinal2.png')

# importing the mode images
modePath = 'resources/modes'
modePathList = os.listdir(modePath)
imgModeList = []
# print(modePathList)
for path in modePathList:
    imgModeList.append(cv.imread(os.path.join(modePath, path)))

# print(len(imgModeList))


detector = FaceDetector()


# 2.graphic part

# 4.Face recognition:
# 4.1 Load the Encoding file
print("Loading encoded file...")
file = open("EncodedFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()


# 4.2 separate studentIds
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encoded file load successfully...")

face_names = []


while True:
    success, img = cap.read()

    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[35:35+640, 835:835+380] = imgModeList[1]
    img, bboxs = detector.findFaces(img)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        cv.circle(img, center, 5, (255, 0, 255), cv.FILLED)
        

    # 4.3 image resizing to 1/4th
    imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    # 4.4 faces and encodings in the current frame
    faceCurrentFrame = face_recognition.face_locations(imgSmall) # locations of the face
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)


    # 4.5 compare generated encodings with existing encodings
    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame): #zip will iterate both list at same time
        # 4.6 find matches
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) # return true/false

        #lower the distance better the match
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        print("matches:", matches)
        print("faceDistance:", faceDis)

        # 4.7 get index of list value
        matchIndex = np.argmin(faceDis) #minimum index value of a list
        print("best Match:", matchIndex)

        # match index 
        if matches[matchIndex]:
            # print("known face detected")
            # print(studentIds[matchIndex])
            name = studentIds[matchIndex]
            face_names.append(name)

            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2- y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0) #rt = rectangle thickness
            
            # # Display the results
            # for (top, right, bottom, left), name in zip(faceCurrentFrame, face_names):
            #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            #     top *= 4
            #     right *= 4
            #     bottom *= 4
            #     left *= 4

            #     # Draw a box around the face
            #     cv.rectangle(img, (left, top), (right, bottom), (255, 0, 255), 1)

            #    # Draw a label with a name below the face
            #     cv.rectangle(img, (left, bottom - 45), (right, bottom), (255, 0, 255), cv.FILLED)
            #     font = cv.FONT_HERSHEY_DUPLEX
            #     cv.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            

            

        # face_names.append(name)



    # cv.imshow("Face Attendance", img)
    cv.imshow("Face Attendance", imgBackground)
    cv.waitKey(1)