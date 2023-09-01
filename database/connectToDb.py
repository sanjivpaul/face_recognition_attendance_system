import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


# 5.1: db setup
cred = credentials.Certificate("database/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    # 5.2: URL of realtime db
    "databaseURL" : "https://faceattendance-rtdb-default-rtdb.firebaseio.com/"
})

# 6.1: create reference of the db
ref = db.reference("")