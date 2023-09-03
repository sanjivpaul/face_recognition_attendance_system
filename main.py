import cv2 as cv
from cvzone.FaceDetectionModule import FaceDetector
import pickle
import face_recognition
import numpy as np
import cvzone
import os
from datetime import datetime
import pyttsx3
import speech_recognition as sr


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


# initialize audio
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[0].id)


def speak(audio):
    '''
    Speak function will Speak our audio string
    '''
    engine.say(audio)
    engine.runAndWait()


# 8.3:update mode type
modeType = 0
counter = 0
id = -1
imgFaculty = []

while True:
    try:
        success, img = cap.read()

        imgBackground[162:162+480, 55:55+640] = img
        # modetype initialize 0
        imgBackground[35:35+640, 835:835+380] = imgModeList[modeType]
        img, bboxs = detector.findFaces(img)

        # if bboxs:
        #     # bboxInfo - "id","bbox","score","center"
        #     center = bboxs[0]["center"]
        #     cv.circle(img, center, 5, (255, 0, 255), cv.FILLED)

        # 4.3 image resizing to 1/4th
        imgS = cv.resize(img, (0, 0), None, 0.25, 0.25)
        imgSmall = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

        # 4.4 faces and encodings in the current frame
        faceCurrentFrame = face_recognition.face_locations(
            imgSmall)  # locations of the face
        encodeCurrentFrame = face_recognition.face_encodings(
            imgSmall, faceCurrentFrame)

        if faceCurrentFrame:
            # 4.5 compare generated encodings with existing encodings
            # zip will iterate both list at same time
            for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
                # 4.6 find matches
                matches = face_recognition.compare_faces(
                    encodeListKnown, encodeFace)  # return true/false

                # lower the distance better the match
                faceDis = face_recognition.face_distance(
                    encodeListKnown, encodeFace)

                # print("matches:", matches)
                # print("faceDistance:", faceDis)

                # 4.7 get index of list value
                # minimum index value of a list
                matchIndex = np.argmin(faceDis)
                # print("best Match:", matchIndex)

                # match index
                if matches[matchIndex]:
                    # print("known face detected")
                    # print(studentIds[matchIndex])
                    id = studentIds[matchIndex]
                    # face_names.append(id)
                    # print(id)

                    # rectangle for face detection
                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(
                        imgBackground, bbox, rt=0)  # rt = rectangle thickness

                    # 8.4: logic for modeType
                    if counter == 0:
                        cvzone.putTextRect(
                            imgBackground, "Loading", (275, 400))
                        cv.imshow("Face Attendance", imgBackground)
                        cv.waitKey(1)
                        counter = 1
                        modeType = 1

            # if counter not 0
            if counter != 0:

                # 8.5.1: first frame if counter == 1 then download & show student info
                if counter == 1:
                    # 9.1 get the info => download all the data
                    # studentInfo = db.reference(f'Students/{id}').get()
                    facultyInfo = db.reference(f"Faculty/{id}").get()
                    # print(studentInfo)
                    print(facultyInfo)

                    # 9.3:get the image from storage fetch image
                    blob = bucket.get_blob(f'images/{id}.png')
                    arr = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgFaculty = cv.imdecode(arr, cv.COLOR_BGRA2BGR)
                    newImgFac = cv.resize(imgFaculty, (216, 216))

                    # 9.4: update data of attendance
                    # datetimeObject = datetime.strptime(
                    #     studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")

                    datetimeObject = datetime.strptime(
                        facultyInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")

                    # 9.5: secondsElapsed = (currentTime - lastAttendanceTime )
                    secondsElapsed = (
                        datetime.now()-datetimeObject).total_seconds()
                    # print(secondsElapsed)

                    # 9.7: after 30 seconds later it will take attendance again
                    if secondsElapsed > 30:

                        # # 9.2:update total attendance
                        # ref = db.reference(f'Students/{id}')
                        # studentInfo['total_attendance'] += 1  # increament by 1
                        # ref.child('total_attendance').set(
                        #     studentInfo['total_attendance'])

                        # 9.2.1:update total attendance of faculty
                        ref = db.reference(f'Faculty/{id}')
                        facultyInfo['total_attendance'] += 1  # increament by 1
                        ref.child('total_attendance').set(
                            facultyInfo['total_attendance'])

                        # 9.6: update last attendance time
                        ref.child("last_attendance_time").set(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[35:35+640, 835:835 +
                                      380] = imgModeList[modeType]

                if modeType != 3:
                    # 8.5.3: update modetype=2 and change the image background
                    if 10 < counter < 20:
                        modeType = 2
                    # show faculty detaila modeType1
                    imgBackground[35:35+640, 835:835 +
                                  380] = imgModeList[modeType]

                    # 8.5.2: display all the details
                    if counter <= 10:
                        # # show details student:
                        # cv.putText(imgBackground, str(studentInfo['total_attendance']), (
                        #     858, 110), cv. FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                        # cv.putText(imgBackground, str(
                        #     studentInfo['class_Roll_No']), (858, 475), cv.    FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                        # cv.putText(imgBackground, str(
                        #     studentInfo['year']), (858, 505), cv. FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                        # cv.putText(imgBackground, str(
                        #     studentInfo['session']), (858, 535), cv.  FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                        # # center name
                        # (w, h), _ = cv.getTextSize(
                        #     studentInfo['name'], cv.FONT_HERSHEY_COMPLEX, 1, 1)
                        # offset = (380-w)//2
                        # cv.putText(imgBackground, str(
                        #     studentInfo['name']), (848+offset, 415), cv.  FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 1)

                        # show faculty details:
                        cv.putText(imgBackground, str(facultyInfo['total_attendance']), (
                            858, 110), cv. FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                        cv.putText(imgBackground, str(
                            facultyInfo['dept']), (858, 475), cv.    FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                        cv.putText(imgBackground, str(
                            facultyInfo['designation']), (858, 505), cv. FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

                        # center name
                        (w, h), _ = cv.getTextSize(
                            facultyInfo['name'], cv.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (380-w)//2
                        cv.putText(imgBackground, str(
                            facultyInfo['name']), (848+offset, 415), cv.  FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 1)

                        # show profile to frame of webcam
                        imgBackground[142:142+216, 915:915+216] = newImgFac

                        facultyGender = facultyInfo['gender']
                        # greeting name
                        hour = int(datetime.now().hour)
                        if hour >= 0 and hour < 12:
                            if facultyGender == 'male':
                                speak(f"Good Morning! {facultyInfo['name']} sir")
                            else:
                                speak(f"Good Morning! {facultyInfo['name']} ma'am")


                        elif hour >= 12 and hour < 18:
                            if facultyGender == 'male':
                                speak(f"Good Aftrnoon! {facultyInfo['name']} sir")
                            else:
                                speak(f"Good Aftrnoon! {facultyInfo['name']} ma'am")

                        else:
                            if facultyGender == 'male':
                                speak(f"Good Evening! {facultyInfo['name']} sir")
                            else:
                                speak(f"Good Evening! {facultyInfo['name']} ma'am")

                        speak("Attendance successfull")

                    counter += 1

                    # reset all the value
                    if counter >= 20:
                        counter = 0
                        modeType = 0
                        studentInfo = []
                        newImgFac = []
                        imgBackground[35:35+640, 835:835 +
                                      380] = imgModeList[modeType]

        else:
            modeType = 0
            counter = 0

        # cv.imshow("Face Attendance", img)
        cv.imshow("Face Attendance", imgBackground)
        cv.waitKey(1)

    except ValueError:
        print("face not found!")
