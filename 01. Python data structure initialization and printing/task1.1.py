#creating tuples
weekTuple = ("Monday", "Tuesday","Wednesday","Thursday","Friday",
             "Saturday","Sunday")

#creating list
weekList = ["Monday", "Tuesday","Wednesday","Thursday","Friday",
             "Saturday","Sunday"]

#creating set
weekSet = { "Monday", "Tuesday","Wednesday","Thursday","Friday",
             "Saturday","Sunday"}

#creating dictonary
weekDict = {
    "Monday" : 1,
    "Tuesday" : 2,
    "Wednesday" : 3,
    "Thursday" : 4, 
    "Friday" : 5,
    "Saturday" : 6,
    "Sunday" : 7
    }

#creating display function for output
def display(data):
    data_type = type(data)              #to get data type
    if data_type == list:
        print("My list : ",data_type)
    elif data_type == tuple:
        print("My tuple : ",data_type)
    elif data_type == set:
        print("My set : ",data_type)
    else :
        print("My dict : ",data_type)
        
    for element in data:
        print (element)

display(weekList);
display(weekTuple);
display(weekSet);
display(weekDict);