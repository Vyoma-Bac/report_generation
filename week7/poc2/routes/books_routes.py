from flask import Blueprint, render_template, request, jsonify,flash
from models.Books import Books
from models.Userdata import Userdata
from sqlalchemy import select,engine,func,update,and_,Integer,cast
from extensions import db

books_routes = Blueprint('books_routes', __name__)

def getbooks() -> list:
    books = []
    stmt = db.session.query(Books).filter(Books.available > 0).all()
    books = stmt
    return books

@books_routes.route('/login/userpage/<uname>/searchbook',methods=["POST","GET"])
def searchbook(uname):
    if request.method == 'POST':
        bname = request.form.get("bname")
        stmt = db.session.query(Books).filter(func.lower(Books.bname) == bname.lower(), Books.available > 0).all()
        if stmt:
            msg = "Book is available"
            return render_template("searchbook.html", res=stmt)
    books = getbooks()
    msg = "Book is not available"
    return render_template("userpage.html", books=books, msg=msg, uname=uname)          

@books_routes.route('/login/userpage/<uname>/issuebook/',methods=["POST","GET"])
def issuebook(uname):
    msg = ''
    a = request.view_args.copy()
    uname = a.get('uname')
    bname = a.get('bname')
    user_query = select(Userdata).where(and_(Userdata.uname == uname, Userdata.banned == 'Active'))
    user_result = db.session.execute(user_query).fetchone()
    if user_result:
        # Check if the user can issue more books
        issued_books_query = select(Userdata).where(and_(~Userdata.issued_books.ilike(f'%{bname}%'),cast(Userdata.issued_books_no, Integer) <= 5))
        issued_books_result = db.session.execute(issued_books_query).fetchall()

        if not issued_books_result:
            # Check if the book is available
            books_query = select(Books).where(and_(Books.bname == bname, Books.available > 0))
            books_result = db.session.execute(books_query).fetchall()

            if books_result:
                # Update the Books table and Userdata table
                stmt = update(Books).where(Books.bname == bname).values(available=Books.available - 1)
                db.session.execute(stmt)

                issued_books_query = select(Userdata.issued_books, Userdata.issued_books_no).where(Userdata.uname == uname)
                issued_books_result = db.session.execute(issued_books_query).fetchone()

                ib = ''
                ibn = 0

                if issued_books_result:
                    ib, ibn = issued_books_result

                if ib is None:
                    ib = bname
                    ibn = 1
                else:
                    ib = f"{ib},{bname}"
                    ibn += 1

                stmt = update(Userdata).where(Userdata.uname == uname).values(issued_books=ib, issued_books_no=ibn)
                db.session.execute(stmt)
                db.session.commit()

                msg = "Book issued successfully"
            else:
                msg = "Book not available"
        else:
            msg = "Cannot issue more books"
    else:
        msg = "User is banned from issuing more books"

    books = getbooks()
    flash(msg, category='success')
    return render_template("userpage.html", uname=uname, books=books, msg=msg)