# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:32:30 2022

@author: David Redondo Quintero
"""

from tkinter import*

if __name__=="__main__":

    window = Tk()
    window.title("Sample Hello World")

    # Defining the button and label properties.
    # What we are actually doing, is just to call a class "button" already
    # defined within the tkinter library
    # All the inputs are the arguments that the constructor needs to create an object of the class
    # It's going to be the same with all the functionalities we need to use to create an interface with this library

    button = Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue", # With these two last sentences we are specifying the colours when one clicks on it
    fg="yellow",
)
    label = Label( text="Hello, Tkinter",
    foreground="white",  # Set the text color to white
    background="black"  # Set the background color to black
    )

    #Getting User input with "Entry" Widegets:
    """
    When you need to get a little bit of text from a user, 
    like a name or an email address, use an Entry widget. 
    It'll display a small text box that the user can type some text into. 
    Creating and styling an Entry widget works pretty much exactly like with Label and Button widgets. 
    For example, the following code creates a widget with a blue background, 
    some yellow text, and a width of 50 text units:
    """
    entry = Entry(fg="yellow", bg="blue", width=50)

    """
    The interesting bit about Entry widgets isn't how to style them, though.  
    There are three main operations that you can perform with Entry widgets.

        - Retrieving text with .get()
        - Deleting text with .delete()
        - Inserting text with .insert()

    These operations can be used in a method such as the method that can be called 
    when the user press a button.
    """
    
    # We need to write the following lines so that widgets are visible

    label.pack()
    button.pack()
    entry.pack()
    window.mainloop()
    