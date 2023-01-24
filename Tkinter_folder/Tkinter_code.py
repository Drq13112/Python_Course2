# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:32:30 2022

@author: David Redondo Quintero
"""

from tkinter import*

class Prueba:
    def __init__(self,ventana):
        self.ventana=ventana
        self.ventana.title("Hello World")
        

if __name__=="__main__":

    ventana = Tk()
    aplicacion = Prueba(ventana)

    button = Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
)
    label = Label( text="Hello, Tkinter",
    foreground="white",  # Set the text color to white
    background="black"  # Set the background color to black
    )

    label.pack()
    button.pack()
    ventana.mainloop()
    