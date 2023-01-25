"""
Here we have a little more complex example with this library,
 despite of that, it shows its potencial.
"""
 
from tkinter import*
  
window=Tk()
window.title("Sample_entry")
 
# setting the windows size
window.geometry("600x400")
  
# declaring string variable
# for storing name and password
name_var=StringVar()
passw_var=StringVar()
 
  
# defining a function that will
# get the name and password and
# print them on the screen
def submit():
 
    name=name_var.get()
    password=passw_var.get()
     
    print("The name is : " + name)
    print("The password is : " + password)
     
    name_var.set("")
    passw_var.set("")
     
     
# creating a label for
# name using widget Label
name_label = Label(window, text = 'Username', font=('calibre',10, 'bold'))
  
# creating a entry for input
# name using widget Entry
name_entry = Entry(window,textvariable = name_var, font=('calibre',10,'normal'))
  
# creating a label for password
passw_label = Label(window, text = 'Password', font = ('calibre',10,'bold'))
  
# creating a entry for password
passw_entry=Entry(window, textvariable = passw_var, font = ('calibre',10,'normal'), show = '*')
  
# creating a button using the widget
# Button that will call the submit function
sub_btn=Button(window,text = 'Submit', command = submit)
  
# placing the label and entry in
# the required position using grid
# method
name_label.grid(row=0,column=0)
name_entry.grid(row=0,column=1)
passw_label.grid(row=1,column=0)
passw_entry.grid(row=1,column=1)
sub_btn.grid(row=2,column=1)
  
# performing an infinite loop
# for the window to display
window.mainloop()