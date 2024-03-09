import streamlit as st
import json
file_path = "library_data.json"
file_path2 = "book_data.json"
try:
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    with open(file_path2, 'r') as json_file:
        book_data = json.load(json_file)
except FileNotFoundError:
    st.warning("Can't Connect")
addbooks=[]

class UserValidation(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class PasswordValidation(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class BookAvailability(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class OverDue(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


corr=0
uname = st.text_input("Enter User Name")
pswd = st.text_input("Password")
if uname and pswd:
    try:
        if uname not in data:
            raise (UserValidation(uname))
        else:
            try:
                if "password" in data[uname]:
                    password = data[uname]['password']
                    corr=2
                else: raise (PasswordValidation(pswd))
                if password == "" or pswd != password:
                    raise (PasswordValidation(pswd))
            except PasswordValidation as error:
                st.warning("Password Incorrect")
    except UserValidation as error:
        st.warning('User does not exist')
bookname = st.selectbox('Select Book',("Operating System Concepts","Database System Concepts","Artificial Intelligence","Computer Architecture"))
if bookname:
    try:
            val = book_data.get(bookname)
            if val <= 0:
                raise (BookAvailability(bookname))
            corr=3
            addbooks.append(bookname)
    except BookAvailability as error:
        st.warning("Book is not available")
if st.button("Issue Book"):
    val = data[uname].get("book_limit")
    ovd = data[uname].get("overdue")
    corr1=0
    try:
        if val > 0:
            corr1 = 1
        else:
            msg="Alredy issued 5 books"
            raise (OverDue(msg))
        if ovd==0 and bookname not in data[uname]["borrowed_books"]:
            corr=2
        else:
            msg="Can't Issue more books"
            raise (OverDue(msg))
    except OverDue as error:
        st.warning(error.value)

    if corr == 3 and corr1==2:
        b_books = data[uname].get("borrowed_books")
        print(b_books)
        data[uname]["book_limit"] = val-1
        data[uname]["borrowed_books"].extend(addbooks)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file,indent=2)
        val=book_data[bookname]
        book_data[bookname]= val-1
        with open(file_path2, 'w') as json_file2:
            json.dump(book_data, json_file2,indent=2)
        st.success("Isuued a book")
