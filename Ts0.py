#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:58:49 2026

@author: joaco
"""

import numpy as np
import matplotlib.pyplot as plt

# Hago la funcion senoidal

def funcion_seno(vmax, dc, ff, ph, nn, fs):
    
    tt = (np.arange(nn)/fs).reshape(-1, 1)
    
    xx = (vmax * np.sin(2 * np.pi * ff * tt + ph) + dc).reshape(-1, 1)
    
    return tt, xx

#Cantidad de muestras
N=1000

#Frecuencia de muestreo 
fs=1000 

# llamo a la funcion y le doy los datos que quiero
tt, xx = funcion_seno(vmax=1, dc=0, ff=1, ph=0, nn= N, fs = fs)

#Grafico la funcion

plt.figure(1)
plt.plot(tt, xx)
plt.title('Funcion seno')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.grid(True)
plt.show()

#Ahora quiero hacer un vector para las frecuencias de prueba
Frecuencias= [500, 999, 1001, 2001]

#Y despues hacer un loop para pasarle esos valores de vector a la funcion para que los grafique
i=2#Este lo uso para cambiar el figure number

for frecNow in Frecuencias:
    tt, xx = funcion_seno(vmax=1, dc=0, ff = frecNow, ph=0, nn= N, fs = 1000)
    #cuantas muestras le pongo?
    
    
    plt.figure(i)
    plt.plot(tt, xx)
    plt.title(f"Frecuencia {frecNow} Hz" )
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (V)')
    plt.grid(True)
    plt.show()
    
    i+= 1
    
#For para poner los 4 graficos juntos y compararlos    
    
plt.figure(6)

posicion = 1

for frecNow in Frecuencias:
    
    tt, xx = funcion_seno(
        vmax=1,
        dc=0,
        ff=frecNow,
        ph=0,
        nn=N,
        fs=1000
    )
    
    plt.subplot(2, 2, posicion)
    plt.plot(tt, xx)
    plt.title(f"Frecuencia {frecNow} Hz")
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (V)')
    plt.grid(True)
    
    posicion += 1

plt.tight_layout()
plt.show()














