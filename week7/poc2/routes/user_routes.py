from flask import Blueprint, render_template, request, jsonify,flash
from models.Books import Books
from models.Userdata import Userdata
user_routes = Blueprint('user_routes', __name__)
from sqlalchemy import select,engine,func,update,and_
from extensions import db

def getbooks() -> list:
    books = []
    stmt = db.session.query(Books).filter(Books.available > 0).all()
    books = stmt
    return books

@user_routes.route('/login/userpage/<uname>/userprofile/update_userdata',methods=["POST","GET"])
def update_userdata(uname):
    books = []
    msg = ''
    uname = request.form.get("uname")
    if request.method == "POST":
        print(uname)
        eid = request.form.get("eid")
        cno = request.form.get("cno")
        addr = request.form.get("addr")
        user_data = db.session.query(Userdata).filter(Userdata.uname == uname).first()
        if user_data:
            # Update user data
            user_data.eid = eid
            user_data.cno = cno
            user_data.addr = addr
            # Commit changes to the database
            db.session.commit()
            books = getbooks()
            flash("Data updated successfully", category='success')
            return render_template("userpage.html", uname=uname, books=books, msg=msg)
        else:
            msg = "Can't update data"
    return render_template("userpage.html", uname=uname, books=books, msg=msg)

@user_routes.route('/login/userpage/<uname>/userprofile',methods=["POST","GET"])
def userprofile(uname):
    uname = request.view_args.get('uname')
    info = db.session.query(Userdata).filter(Userdata.uname == uname).all()
    return render_template("userprofile.html", uname=uname, info=info)

@user_routes.route('/login/userpage/<uname>',methods=["POST","GET"])
def userpage(uname):
    a = request.view_args.copy()
    uname = a.get('uname')
    books=getbooks()
    return render_template("userpage.html",uname=uname,books=books)
