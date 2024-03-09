from extensions import db

class Userdata(db.Model):
    __tablename__ = "userdata"
    id = db.Column(db.Integer, primary_key=True)
    uname = db.Column(db.String(200), unique=True, nullable=False)
    banned = db.Column(db.String(6), default='Active')
    issued_books_no = db.Column(db.Integer, unique=False, nullable=True, default=0)
    eid = db.Column(db.String(40), unique=False, nullable=True)
    cno = db.Column(db.String(10), unique=False, nullable=True)
    addr = db.Column(db.String(200), unique=False, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    user_relation = db.relationship("User", back_populates="userdata")
    issued_books = db.Column(db.String(400), unique=False, nullable=True)
