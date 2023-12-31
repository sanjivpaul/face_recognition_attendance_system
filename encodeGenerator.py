import cv2 as cv
import face_recognition
import pickle
import os
# 7.1:for store images into db storage
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# 7.2: db & storage setup
cred = credentials.Certificate("database/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    # 5.2: URL of realtime db
    "databaseURL": "https://faceattendance-rtdb-default-rtdb.firebaseio.com/",
    "storageBucket":"faceattendance-rtdb.appspot.com"
})


# importing student images
imagePath = "images"  # path of image diretory
imagePathList = os.listdir(imagePath)  # making list of image directory
# print(imagePathList)

imgList = []  # initialize a empty array

# extract img id
studentIds = []

for path in imagePathList:
    imgList.append(cv.imread(os.path.join(imagePath, path)))
    # print(path);
    # print(os.path.splitext(path)[0]) # split student id
    studentIds.append(os.path.splitext(path)[0])

    # 7.3: add storage path
    fileName = f'{imagePath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName) #upload images
    # print("image uploaded successfully")


# print(len(imgList));
print(studentIds)


# encode images
# step 1: change the color (BGR to RGB)
# step 2: find the encoding
def findEncoding(imagesList):
    encodeList = []  # empty array

    for img in imagesList:  # for each image of imgList
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding started...")
encodeListKnown = findEncoding(imgList)

# which studentIds belongs to which encoding
encodeListKnownWithIds = [encodeListKnown, studentIds]

print(encodeListKnown)
print("Encoding completed")

# pickling of encodingListKnownWithIds data in new file
file = open("EncodedFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File save successfully:)")
