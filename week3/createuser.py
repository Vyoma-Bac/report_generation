import streamlit as st
import re
regex = re.compile(r'\b[@_!#$%^&*()<>?/\|}{~:]')
st.title("User Login")
fname=st.text_input("Enter First Name")
lname=st.text_input("Enter Last Name")
uname=st.text_input("Username")
if uname:
    if not (len(uname)>=6 and bool(re.search(r"\d" and r"\b[a-zA-Z]+",uname))):
        uname=''
        st.warning("Not valid username")
        

pswd=st.text_input("Password")
if len(pswd)>=8:
        a=bool(any(ele.isupper() for ele in pswd))
        if bool(a and re.search(r"\d",pswd) and re.search(r"\b[a-z]+",pswd) and regex.search(pswd)):
           print("Valid")
else:
    pswd=''
    st.warning("Not valid password")

            
age=st.number_input("Age")
if age<=18:
    age=''
    st.warning("Age should be greater than 18")
if st.button("Create Account"):
    try:
            f=open("hello.txt",'r')
            text_data = f.read()
            i=text_data.find(uname)
            print(int(i))
            f.close()
    except FileNotFoundError:
            f = open('hello.txt', 'x')
    if i!=-1:
        st.warning("Username already taken")
    if fname and lname and uname and pswd and age:
        
        line = '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(fname, lname, uname,pswd ,age)
        print(line)
        try:
            f = open('hello.txt', 'a')
        except FileNotFoundError:
            f = open('hello.txt', 'x')
        f.write(line)
    else:
        st.warning("Can't create account") 




