class InventoryItem:
   def __init__(self, name, quantity, price):
       self.name,self.quantity,self.price = name,quantity,price

   def display_info(self):
       print(f"{self.name} - Quantity: {self.quantity}, Price: ${self.price:.2f}")
    
inventory = [
 InventoryItem("Laptop", 10, 800.00),
       InventoryItem("Mouse", 50, 20.00),
       InventoryItem("Keyboard", 30, 50.00)
   ]

def sell_Item():
    name = input("Enter item name to sell: ")
    for item in inventory:
               if item.name.lower() == name.lower():
                   sell_quantity = int(input("Enter quantity to sell: "))
                   if sell_quantity <= item.quantity:
                       item.quantity -= sell_quantity
                       total_price = sell_quantity * item.price
                       print(f"{sell_quantity} {item.name}(s) sold for ${total_price:.2f}")
                   else:
                       print("Insufficient quantity in stock.")
                   break
    else:
               print(f"{name} not found in the inventory.")



def main():
 
   while True:
       print("\nInventory Management System\n1. Display Inventory\n2. Add Item\n3. Sell Item\n4. Exit")
       choice = input("Enter your choice (1-4): ")

       if choice == "1":
           print("\nCurrent Inventory:")
           for item in inventory:item.display_info()

       elif choice == "2":
           name,quantity,price = input("Enter item name: "),int(input("Enter quantity: ")), float(input("Enter price per item: "))
           inventory.append(InventoryItem(name, quantity, price))
           print(f"{name} added to the inventory.")

       elif choice == "3":
           sell_Item()

       elif choice == "4":
           print("Exiting Inventory Management System.")
           break

       else:
           print("Invalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
   main()
