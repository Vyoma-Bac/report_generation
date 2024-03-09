from extensions import db

class Books(db.Model):
    __tablename__ = "books"
    id = db.Column(db.Integer, primary_key=True)
    bname = db.Column(db.String(200), unique=True, nullable=True)
    available = db.Column(db.Integer)
    author = db.Column(db.String(200), unique=False, nullable=True)
    genre = db.Column(db.String(200), unique=False, nullable=True)
    summary = db.Column(db.String(600), unique=False, nullable=True)
