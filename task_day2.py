#Restaurant Menu App: Use tuples to store menu items (name, price) and lists to store ingredients. Allow users to view menu, search for items, and calculate bill. Involve type casting for user input validation.

#("Caprese Salad",120),("Paneer Tikka",160),("Lentil Soup",90),
#("Veggie  Wrap",130),("Aloo Gobi",140),("Mushroom Risotto",190),("Palak Paneer",180),("Spring Rolls",120),("Noodles with Pesto",200)

dish=(("Veggie Burger",150),("Margherita Pizza",200),("Biryani",220),("Paneer Tikka",20),("Veggie Wrap",100),("Spring Rolls",120))
total1=0
ingredients=["Veggie patty, Lettuce,Pickles, Cheese, Condiments (mayonnaise, ketchup)","Tomato sauce, Fresh mozzarella cheese, Basil leaves","Mixed vegetables (such as carrots, peas, beans),Yogurt, Spices (cumin, coriander, cardamom, cloves)","Paneer,Yogurt,Spices (cumin, coriander, garam masala),Bell peppers","Tortilla wrap,Hummus,Mixed vegetables (lettuce, tomato, cucumber, bell peppers),Feta cheese","Spring roll wrappers,Vegetables,Bean sprouts,Soy sauce"]
ord=[]
def view_menu():
    i=0
    for d,p in dish:
        print(d,p)
        print(ingredients[i])
        i+=1
def search_dish():
        dname=input("Enter Dish or price:")
        a=[]
        if dname.isdigit():
            price=int(dname)
            a=[(d,p) for d,p in dish if p==price or p<price]
        else:
           a=[(d,p) for d,p in dish if d==dname]
        if a:
            for d,p in a:
                 print(d,p)
        else:print("No dishes found")
def order_dish():
     a=input("Enter dish name:")
     ord=(a.split(","))
     
     return ord
def generate_bill(ord):
     name=input("Enter Name:")
     cno=str(input("Enter Conatct no:"))
     if ord:
         total=0  
         for i in ord:
            for d,p in dish:
                if i.lower()==d.lower():
                    total=total+p
     bill={"Name":name,"Contact no:":cno,"Orders":ord,"Bill(Excluding GST):":total}
     print(bill)
     wt_gst=int(input("Press 1 to get Bill with GST:"))
     if wt_gst==1:
           bill_wt_gst={"Name":name,"Contact no:":cno,"Orders":ord,"Bill(Including GST):":total+total*(18/100)}
           print(bill_wt_gst)
     
def exit_app():
     exit()
op='y'
while op=='y':
    print("\nEnter 1 to view menu\nEnter 2 to search available Dishes\nEnter 3 to order the dish\nEnter 4 to generate bill\nEnter 5 to Exit\n")
            
    a=int((input("Choose an option:")))
    if a==1:
            view_menu()  
    elif a==2:
            search_dish() 
    elif a==3:
            ord=order_dish()
    elif a==4:
            generate_bill(ord)
    elif a==5:
            exit_app()
    else:
                print("Invalid")
    op=input("Enter Y to continue:")
        

        
        
