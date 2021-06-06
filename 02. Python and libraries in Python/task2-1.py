"""
@author: Kovarthanan K
"""

def find_factorial():
    
    #asigning number to negative value for enable while loop
    number = -1 
    while(number < 0):
         number = int(input("Please input a nonnegative integer : ")) 
         
    factorial_output = 1
    
    if(number==0 or number==1):
        #when the input is 0 or 1
        print("Factorial of {} : {}".format(number, factorial_output))
    else : 
        #when input is more then 1   
        for i in range(2,number+1):
            factorial_output *= i
        print("Factorial of {} : {}".format(number, factorial_output))
        
#calling find_factorial function to find factorial
find_factorial()