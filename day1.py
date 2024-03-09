"""- Given a list of student names, create a new list containing only the names longer than 7 characters. 
input = ["Alice", "Bob", "Charlie", "David", "Emily", "Frank", "Grace"]
output - ['Alice', 'Charlie', 'Emily', 'Frank', 'Grace']

Define a function that takes a list of numbers and returns a new list containing the squares of the even numbers and the cubes of the odd numbers. 

input = [1, 2, 3, 4, 5, 6]
output -  [1, 4, 9, 16, 25, 36]

- Create a function that takes two tuples of the same length and returns a new tuple where each element is the sum of the corresponding elements in the original tuples. 

input - 
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)

output - (5, 7, 9)


- You are given a list of students, where each student is represented by a tuple containing their name and age. Write a Python program to convert the list of tuples into a list of dictionaries, where each dictionary contains the 'name' and 'age' as keys.

input - students_tuples = [("Alice", 22), ("Bob", 25), ("Charlie", 20)]
output - [{'name': 'Alice', 'age': 22}, {'name': 'Bob', 'age': 25}, {'name': 'Charlie', 'age': 20}]

- You have a list of integers, and you want to create a set containing only the prime numbers.

input - [2, 3, 5, 7, 8, 11, 13, 14, 17, 19, 20]
output - {2, 3, 5, 7, 11, 13, 17, 19}


- You have a list of words, and you want to create a set containing only the words that have all distinct characters.

input - ['hello', 'world', 'python', 'unique', 'set', 'comprehension']
output - {'world', 'unique', 'set', 'comprehension'}


- You have a list of strings representing names, and you want to create a dictionary where the keys are the first letters of the names, and the values are lists of names starting with that letter.


input -  ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Anna']
output - {'A': ['Alice', 'Anna'], 'B': ['Bob'], 'C': ['Charlie'], 'D': ['David'], 'E': ['Eva'], 'F': ['Frank']}"""
"""
input = ["Alice", "Bob", "Charlie", "David", "Emily", "Frank", "Grace"]
a=[name for name in input if len(name)>5]
print(a)

input = [1, 2, 3, 4, 5, 6]
a=[val*val for val in input]
print(a)

input = [2, 3, 5, 7, 8, 11, 13, 14, 17, 19, 20]
def primeno(x):
    if x<2:
        return False
    for i in range(2,x):
        if x%i ==0:
            return False
    return True
a=[val for val in input if primeno(val)]
print(set(a))

input =  ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Anna']
dict={val1[0]:[val for val in input if val1.startswith(val[0])] for val1 in input}
print(dict)



students_tuples = [("Alice", 22), ("Bob", 25), ("Charlie", 20)]
dict= [{'name': name, 'age': age} for name, age in students_tuples]
print(dict)

"""
from datetime import datetime

# Example with the standard date and time format
date_str = '2023-02-28 14:30:00'
date_format = '%Y-%m-%d %H:%M:%S'

date_obj = datetime.strptime(date_str, date_format)
print(date_obj)

# Example with a different format

date_str = '02/28/2023 02:30 PM'
date_format = '%m/%d/%Y %I:%M %p'

date_obj = datetime.strptime(date_str, date_format)
print(date_obj)