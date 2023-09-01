import cv2 as cv
from cvzone.FaceDetectionModule import FaceDetector
import pickle
import face_recognition
import numpy as np
import cvzone
import os


# 8.1: for real time database update
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# 8.2: db & storage setup
cred = credentials.Certificate("database/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    # 5.2: URL of realtime db
    "databaseURL": "https://faceattendance-rtdb-default-rtdb.firebaseio.com/",
    "storageBucket": "faceattendance-rtdb.appspot.com"
})

# for get img from storage
bucket = storage.bucket()

cap = cv.VideoCapture(0)
detector = FaceDetector()

# 2.graphic part
cap.set(3, 640)  # width
cap.set(4, 480)  # height
imgBackground = cv.imread('resources/bg.png')

# importing the mode images
modePath = 'resources/modes'
modePathList = os.listdir(modePath)
imgModeList = []
# print(modePathList)
for path in modePathList:
    imgModeList.append(cv.imread(os.path.join(modePath, path)))


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

# 8.3:update mode type
modeType = 0
counter = 0
id = -1
imgFaculty = []

while True:
    success, img = cap.read()

    imgBackground[162:162+480, 55:55+640] = img
    # modetype initialize 0
    imgBackground[35:35+640, 835:835+380] = imgModeList[modeType]
    img, bboxs = detector.findFaces(img)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        cv.circle(img, center, 5, (255, 0, 255), cv.FILLED)

    # 4.3 image resizing to 1/4th
    imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    # 4.4 faces and encodings in the current frame
    faceCurrentFrame = face_recognition.face_locations(
        imgSmall)  # locations of the face
    encodeCurrentFrame = face_recognition.face_encodings(
        imgSmall, faceCurrentFrame)

    # 4.5 compare generated encodings with existing encodings
    # zip will iterate both list at same time
    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        # 4.6 find matches
        matches = face_recognition.compare_faces(
            encodeListKnown, encodeFace)  # return true/false

        # lower the distance better the match
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        print("matches:", matches)
        print("faceDistance:", faceDis)

        # 4.7 get index of list value
        matchIndex = np.argmin(faceDis)  # minimum index value of a list
        print("best Match:", matchIndex)

        # match index
        if matches[matchIndex]:
            # print("known face detected")
            # print(studentIds[matchIndex])
            id = studentIds[matchIndex]
            # face_names.append(id)
            print(id)

            # rectangle for face detection
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(
                imgBackground, bbox, rt=0)  # rt = rectangle thickness

            # 8.4: logic for modeType
            if counter == 0:
                counter = 1
                modeType = 1

    # if counter not 0
    if counter != 0:

        # 8.5: first frame if counter == 1 then download & show student info
        if counter == 1:
            # get the info
            studentInfo = db.reference(f'Students/{id}').get()
            print(studentInfo)

            # get the image from storage
            blob = bucket.get_blob(f'images/{id}.png')
            arr = np.frombuffer(blob.download_as_string(), np.uint8)
            imgFaculty = cv.imdecode(arr, cv.COLOR_BGRA2BGR)
            newImgFac = cv.resize(imgFaculty, (216, 216))

            # update data of attaindence 
            

        cv.putText(imgBackground, str(studentInfo['total_attendance']), (858, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        cv.putText(imgBackground, str(studentInfo['class_Roll_No']), (858, 475), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv.putText(imgBackground, str(studentInfo['year']), (858, 505), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv.putText(imgBackground, str(studentInfo['session']), (858, 535), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        # center name
        (w, h), _ = cv.getTextSize(studentInfo['name'], cv.FONT_HERSHEY_COMPLEX, 1, 1)
        offset = (380-w)//2
        cv.putText(imgBackground, str(studentInfo['name']), (848+offset, 415), cv.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 1)

        # show profile to frame
        imgBackground[142:142+216, 915:915+216] = newImgFac

        counter += 1

        # face_names.append(name)

    # cv.imshow("Face Attendance", img)
    cv.imshow("Face Attendance", imgBackground)
    cv.waitKey(1)
