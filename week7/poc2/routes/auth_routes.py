from flask import Blueprint, render_template, request,redirect,flash,url_for
from models.User import User
from models.Books import Books
from extensions import db
from forms.loginForm import loginForm
from forms.registerForm import registerForm
from flask_bcrypt import Bcrypt

bcrypt=Bcrypt()
auth_routes = Blueprint('auth_routes', __name__)

def getbooks() -> list:
    books = []
    stmt = db.session.query(Books).filter(Books.available > 0).all()
    books = stmt
    return books

@auth_routes.route('/register', methods=['GET', 'POST'])
def register():
    form = registerForm(request.form)
    error = None
    if request.method == 'POST' and form.validate():
        hashed_pwd = bcrypt.generate_password_hash(form.psw.data).decode('utf-8')
        existing_user = db.session.query(User).filter_by(uname=form.uname.data).first()
        if existing_user:
            error = "Username already exists"
            return render_template('register.html', error=error, form=form)
        user = User(uname=form.uname.data, pwd=hashed_pwd)
        db.session.add(user)
        db.session.commit()
        flash('Thanks for registering')
        return redirect(url_for('auth_routes.login'))
    return render_template('register.html', form=form, error=error)

@auth_routes.route('/', methods=["POST", "GET"])
def login():
    error = None
    form = loginForm(request.form)
    books = getbooks()
    if request.method == "POST" and form.validate():
        user = db.session.query(User).filter(User.uname == form.uname.data).first()
        if user and bcrypt.check_password_hash(user.pwd, form.pwd.data):
            flash("You are successfully logged in", category='success')
            return redirect(url_for('user_routes.userpage', uname=form.uname.data, books=books))
        else:
            error = "Invalid username or password"
    return render_template("login.html", error=error, form=form)

@auth_routes.route('/login/userpage/<uname>/logout',methods=["POST","GET"])
def logout():
    msg="logged out successfully"
    return redirect(url_for('auth_routes.login',error=msg))
