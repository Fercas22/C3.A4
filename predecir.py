import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from PIL import ImageTk, Image
from tkinter import messagebox

# Enlase para obtner las imagenes: 
# test : https://drive.google.com/file/d/1Ee5jvgBu5eCeUDDg7FcC6QvSnB0_Wohn/view?usp=sharing
# train : https://drive.google.com/file/d/1WW3wNFyo3gtq9VzUt_Gvl-rYNaNvonG1/view?usp=sharing

longitud, altura = 150, 150
modelo = './models/modelo.h5'
pesos_modelo = './models/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)

  return answer

def archivo():
  filename = askopenfilename() 
  print(filename)
  # tk.PhotoImage(file=filename)

  valor = predict(filename)
  if valor==0:
    messagebox.showinfo(message="Pantalon", title="Aviso")
  if valor==1:
    messagebox.showinfo(message="Playera", title="Aviso")
  if valor==2:
    messagebox.showinfo(message="Short", title="Aviso")
  if valor==3:
    messagebox.showinfo(message="Tenis", title="Aviso")
  


root = tk.Tk()
root.config(width=500, height=200)
root.title("Clasificador de imagenes")
boton = ttk.Button(text="Seleccionar imagen", command=archivo)
boton.place(x=200, y=80)

root.mainloop()