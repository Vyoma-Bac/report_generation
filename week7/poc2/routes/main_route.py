from flask import Blueprint, render_template,url_for

main_route = Blueprint('main_route', __name__)

@main_route.route('/')
def index():
    return render_template('login.html') 