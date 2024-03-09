from extensions import db

class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    uname = db.Column(db.String(50), unique=True, nullable=True)
    pwd = db.Column(db.String(300), unique=False, nullable=True)
    userdata = db.relationship('Userdata', backref="user")
