from flask import Flask
from extensions import db
from routes.auth_routes import auth_routes
from routes.main_route import main_route
from routes.user_routes import user_routes
from routes.books_routes import books_routes
def create_app():
    app = Flask(__name__)
    app.secret_key = '012#!ApaAjaBoleh)(*^%'
    # Database Configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:abc123@127.0.0.1:5432/flask'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize Flask-SQLAlchemy extension
    db.init_app(app)

    # Register Blueprints
    
    app.register_blueprint(auth_routes)
    app.register_blueprint(main_route)
    app.register_blueprint(user_routes)
    app.register_blueprint(books_routes)
    return app