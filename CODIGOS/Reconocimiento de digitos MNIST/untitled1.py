# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:44:38 2023

@author: pjime
"""
from PIL import Image

def convertL(nombre):
    ruta_imagen= nombre
    image = Image.open(ruta_imagen).convert('L')
return image

