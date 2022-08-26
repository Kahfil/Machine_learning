largest = None
smallest = None
while True:
    num = input("Enter a number: ")
    if num == "done":
        break
    try :
        nam=int(num)
    except:
        print("Invalid input")


    if largest is None:
        largest=nam
    elif nam>largest:
        largest=nam
    if smallest is None:
        smallest=nam
    elif nam<smallest:
        smallest=nam

print("Maximum is", largest)
print("Minimum is", smallest)
