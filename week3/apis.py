import requests as re
print("End Points:\n1) users\n2) /users/user_id/\n3) users/user_id/enrolments/\n4) users/user_id/institutions/\n5) courses")
ep=input("Enter end point:")
if ep=="users":
    pname=input("Enter Profile name:")
    s=f"users/?profile_name={pname}&"
elif ep=="user_id":
    uid=input("Enter User ID:")
    s=f"users/{uid}/?"
elif ep=="enrolments":
    uid=input("Enter User ID:")
    s=f"users/{uid}/enrolments/?"
elif ep=="institutions":
    uid=input("Enter User ID:")
    s=f"users/{uid}/institutions/?"
elif ep=="courses":
    s=f"courses/?"
else:
    print("Enter valid End point")

url1=f'https://api.openlearning.com/v2.1/{s}api_key=65c34a26fdd3df2861753bd0.fd7c350a314765c8328f1140fe6b71cf50e5ea06b000c245dfd5c9492d2e5ff6'
res=re.get(url1)




print(res.json())
print("Status code:",res.status_code)