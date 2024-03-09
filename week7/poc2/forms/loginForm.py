from wtforms import Form,BooleanField, StringField, PasswordField, TextAreaField, validators,IntegerField

class loginForm(Form):
    uname=StringField('Username',[validators.Length(min=4, max=25)])
    pwd = PasswordField('Password',[validators.DataRequired()])
   