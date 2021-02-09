from Polygon import *
class Triangle(Polygon): 
  
    # overriding abstract method 
    def noofsides(self): 
        print("I have 3 sides") 

R = Triangle() 
R.noofsides() 