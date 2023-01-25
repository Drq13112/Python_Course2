"""
In this code, we'll program a code which allows us to select
an option between some.
"""


# Importing Tkinter module
from tkinter import *
from tkinter.ttk import *

# Creating master Tkinter window
window = Tk()
window.title("Selection button")
window.geometry("175x175")

# Tkinter string variable
# It's able to store any string value
v = StringVar(window, "1")

# Dictionary to create multiple buttons
values = {"Button 1" : "1",
		"Button 2" : "2",
		"Button 3" : "3",
		"Button 4" : "4",
		"Button 5" : "5"}

# Loop is used to create multiple buttons
# rather than creating each button separately
for (text, value) in values.items():
	Radiobutton(window, text = text, variable = v,
		value = value).pack(side = TOP, ipady = 5)

# Infinite loop can be terminated by
# keyboard or mouse interrupt
# or by any predefined function (destroy())
window.mainloop()
