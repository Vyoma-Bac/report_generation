from wtforms import Form,BooleanField, StringField, PasswordField, TextAreaField, validators,IntegerField

class registerForm(Form):
    uname=StringField('Username',[validators.Length(min=4, max=25),validators.DataRequired()])
    eid = StringField('Email Address', [validators.Length(min=6, max=35),validators.DataRequired()])
    psw = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')
    addr = TextAreaField(u'Address', [validators.optional(), validators.length(max=200)])
    cno = TextAreaField('Contact No', [validators.optional(), validators.length(max=10)])
# Create tables
    
