import tkinter as tk  
  
def getScreenSize():  
    root = tk.Tk()  
    width = root.winfo_screenwidth()  
    height = root.winfo_screenheight()  
    root.destroy()  
    return width, height  