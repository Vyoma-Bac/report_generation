import re

doc1 = "asf dob: 1/5/1991 in tech"
doc2 = "DOB 5-5-1991 and educ: 1/6/2010"
doc3 = "hi I am good in python. my date of birth is 6/5/2005"
doc4 = "edu: 1/2/2008 and DOB 5/8/1991 and education 1/6/2010"
doc5 = "skill master overall. my birth 7/02/22"
doc_list=[doc1,doc2,doc3,doc4,doc5]
#for i in doc_list:
    #a=re.findall(r'\b\d{1,2}[/]\d{1,2}[/]\d{2,4}',i) 
    #print(a)

"""
list1 = ["abc.csv","asf.csv.csv","txt-csv.csv","xut.txt"]
for a in list1:
    g=re.split(".[a-z]+$",a)
    print(g[0])"""