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
ref = db.reference("Students")

data = {
     "MCA00122": {
        "name": "Elon Musk",
        "class_Roll_No": "MCA/009/22",
        "session": "2022 - 2024",
        "total_attendance": 13,
        "year": 1,
        "last_attendance_time": "2023-08-29 10:15:00"
    },
    "MCA00922": {
        "name": "Barkha Keshri",
        "class_Roll_No": "MCA/009/22",
        "session": "2022 - 2024",
        "total_attendance": 13,
        "year": 1,
        "last_attendance_time": "2023-08-29 10:15:00"
    },

    "MCA02922": {
        "name": "Sanjiv Paul",
        "class_Roll_No": "MCA/029/22",
        "session": "2022 - 2024",
        "total_attendance": 12,
        "year": 1,
        "last_attendance_time": "2023-08-29 10:15:00"
    },
}

# 6.2: send or insert data into db
for key, value in data.items():
    ref.child(key).set(value) 

print("data inserted successfully!..")
