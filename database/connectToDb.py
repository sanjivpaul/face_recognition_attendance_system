import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


# 5.1: db setup
cred = credentials.Certificate("database/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    # 5.2: URL of realtime db
    "databaseURL": "https://faceattendance-rtdb-default-rtdb.firebaseio.com/"
})

# 6.1: create reference of the db
# ref = db.reference("Students")
ref = db.reference("Faculty")

# data = {
#      "MCA00122": {
#         "name": "Elon Musk",
#         "class_Roll_No": "MCA/001/22",
#         "session": "2022 - 2024",
#         "total_attendance": 13,
#         "year": 1,
#         "last_attendance_time": "2023-08-29 10:15:00",
#         "gender":"male"
#     },
#     "MCA00922": {
#         "name": "Barkha Keshri",
#         "class_Roll_No": "MCA/009/22",
#         "session": "2022 - 2024",
#         "total_attendance": 13,
#         "year": 1,
#         "last_attendance_time": "2023-08-29 10:15:00",
#         "gender":"female"
#     },

#     "MCA02922": {
#         "name": "Sanjiv Paul",
#         "class_Roll_No": "MCA/029/22",
#         "session": "2022 - 2024",
#         "total_attendance": 12,
#         "year": 1,
#         "last_attendance_time": "2023-08-29 10:15:00",
#         "gender":"male"
#     },
# }

# faculty data
data = {
    "MCA01" : {
        "name": "Yogendra Kumar",
        "total_attendance":5,
        "dept":"MCA",
        "designation":"HOD and Assistant Professor",
        "last_attendance_time": "2023-08-29 10:15:00",
        "gender":"male"
    },
    "MCA02" : {
        "name": "Kishore Kumar Ray",
        "total_attendance":5,
        "dept":"MCA",
        "designation":"HOD and Assistant Professor",
        "last_attendance_time": "2023-08-29 10:15:00",
        "gender":"male"
    },

}


# 6.2: send or insert data into db
for key, value in data.items():
    ref.child(key).set(value) 

print("data inserted successfully!..")
