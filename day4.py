#(123) 456-7890
"""phone_numbers = [
    "(123) 456-7890",
    "123-456-7891",
    "123 456 7892",
    "123.456.7893"
]
sol
cleaned_numbers = []
for number in phone_numbers:
    # Remove unwanted characters
    number = number.replace("(", "").replace(")", "").replace("-", "").replace(".", "").replace(" ", "")

    area_code = number[:3]
    prefix = number[3:6]
    line_number = number[6:]

    formatted_number = f"({area_code}) {prefix}-{line_number}"

    cleaned_numbers.append(formatted_number)


print(cleaned_numbers)"""
"""input = 'pythonista'
Output = "Inoh"
a=input[3:-3]
print(a[::-1].capitalize())
"""


a=["https://www.example.com/search?q=python&category=programming&sort=popularity",
"http://blog.example.org/articles?tag=technology&page=2",
"https://api.example.net/data?format=json&apikey=abc123",
"https://www.sample-site.com/products?id=123&category=electronics",
"https://forum.example.org/topic?title=url+parsing&user=admin&sort=latest",
"https://sub.example.com/path/to/page?param1=value1&param2=value2",
"https://www.testing-site.org/search?query=test+case&filter=latest",
"http://www.product-site.com/product-details?id=9876&lang=en",
"https://app.example.io/dashboard?user=admin&theme=dark",
"https://www.news-site.org/articles?category=world&tag=breaking-news&page=1",
"https://forum.sample.org/post?id=456&user=member1&sortBy=votes",
"https://www.travel-site.com/destination?city=paris&lang=fr&date=2024-03-01",
"https://api.testing.net/data?format=xml&token=xyz987&user=admin",
"https://www.weather-site.org/forecast?location=london&units=celsius",
"https://www.blog-example.com/post?id=789&tag=python&author=john_doe"
]
for a in a:
    print("Link:",a)
    temp=a.split(':')
    protocol=temp[0]
    print("Protocol:",protocol)
    temp=a.split('//')
    d_pos=temp[1].find('/')
    print("Domain:",temp[1][0:d_pos])
    subd=temp[1][0:d_pos].split(".")
    if len(subd)==3:
            print("Sub Domain:",subd[0])
    path_val=temp[1][d_pos:].split('?')
    print("Path:",path_val[0])
    q=path_val[1]
    q_para=q.split("&")
    print("Query String:",q_para)
    t_q=''
    temp=str(q_para)
    if temp.startswith("['q"):
        for a in q_para:
                u=a.replace('sort','date')
                t_q=t_q+u+"&"
        new_q=t_q[:-1]
        print("Updated query",new_q)

# slicing problems

input1='program'
print(input1[1::2])
s = 'coder'
# print(s[::0]) #  op => error slice step cant be zero
s = 'doubled'
print(s[1:6][1:3]) # op=>ub
input1='question'
a=input1[1::2]
print(a[::-1])

Input='mirage'
print(Input[2:3]+Input[0:1]+Input[4:5])

s = 'program'
print(s[::2]) # op=>porm

s = 'doubled'
print(s[1:6][1:3]) # op=>ub





