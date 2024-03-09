class InventoryItem:
   def __init__(self, name, quantity, price):
       self.name = name
       self.quantity = quantity
       self.price = price


   def display_info(self):
       print(f"{self.name} - Quantity: {self.quantity}, Price: ${self.price:.2f}")


def main():
   inventory = [
 InventoryItem("Laptop", 10, 800.00),
       InventoryItem("Mouse", 50, 20.00),
       InventoryItem("Keyboard", 30, 50.00)
   ]


   while True:
       print("\nInventory Management System")
       print("1. Display Inventory")
       print("2. Add Item")
       print("3. Sell Item")
       print("4. Exit")


       choice = input("Enter your choice (1-4): ")


       if choice == "1":
           print("\nCurrent Inventory:")
           for item in inventory:
               item.display_info()


       elif choice == "2":
           name = input("Enter item name: ")
           quantity = int(input("Enter quantity: "))
           price = float(input("Enter price per item: "))
           new_item = InventoryItem(name, quantity, price)
           inventory.append(new_item)
           print(f"{name} added to the inventory.")


       elif choice == "3":
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


       elif choice == "4":
           print("Exiting Inventory Management System.")
           break


       else:
           print("Invalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
   main()
