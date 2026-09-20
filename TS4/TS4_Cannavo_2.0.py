#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:11:26 2026

@author: joaco
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sg

#%% Variables

a0 = np.sqrt(2)
N = 1000
omega0 = 250 #en realidad es pi sobre cuatro pero func
Na = 1 #Reemplazar por ruido
fr =  1#cambiarlo por ruido
fs = 1000
R = 200 # Itereaciones del experimento
#varianza = 0.1 aca habria que ver si es esta o la otra

limiteUniforme = 2
P_s = 1 #Potencia
snr_db = 10 #Decibeles
varianza = P_s / (10**(snr_db / 10))
sigma = np.sqrt(varianza)

#%% Condicionando las variables

# Hacer las de ruido
Na = np.random.normal(0.0, sigma, (N, R))
# Hecho matriz para poder sumarlo a la funcion sen

fr = np.random.uniform(-limiteUniforme, limiteUniforme, R)
# 200 por la cantidad de iteraciones de expermiento



# Matrices
n_col = np.arange(N).reshape(N, 1)

#omega1 = omega0 + (fr*(2*(np.pi) / N))

omega1 = 2 * np.pi * (omega0 + fr) / N #Cual de los dos estara bien???
# Vos a este lo multiplicas todo por pi por el hecho de que tiene que star en radianes?

omega1 = omega1.reshape(1, R)
# no hace falta el arange en este

#%% Creacion de funciones

# Eje de frecuencia
ff_vector = np.fft.fftfreq(N, d=1 / fs)
ff_vector_pos = ff_vector[:N//2]

# Hago mi funcion
f_sen = a0 *np.sin(omega1 * n_col) + Na


#FFt
f_fft = (1/N) * np.fft.fft(f_sen, axis=0)

# Formula de potencia --> elevo al cuadrado y divido por 2 para tener Watts
potencia_lineal = (np.abs(f_fft[:N // 2, :]))**2
 
dens_pot = 10 * np.log10(potencia_lineal + 1e-12)

# Estimador de frecuencia
indice_pico = np.argmax(np.abs(f_fft[:N // 2, :]), axis=0)

om_f = ff_vector_pos[indice_pico]

# Estimador de amplitud
a_1_2= 2 * np.abs(f_fft[indice_pico, np.arange(R)])


#%% Ventanas

#La rectangular ya esta hecha pq es multiplicado por uno

####################################################
#FLATTOP
####################################################

# Genero una funcion que sea la multiplicada por mi ventana
f_flattop =  f_sen * sg.flattop(N).reshape(N, 1)
#Por que se le pone el reshape a este

f_flattop_fft = (1/N) * np.fft.fft(f_flattop, axis=0)

# Formula de potencia --> elevo al cuadrado y divido por 2 para tener Watts
potencia_lineal_flattop = (np.abs(f_flattop_fft[:N // 2, :]))**2
 
dens_pot_flattop = 10 * np.log10(potencia_lineal_flattop + 1e-12)

# Estimador de amplitud
indice_pico_flattop= np.argmax(np.abs(f_flattop_fft[:N // 2, :]), axis=0)

a_1_2_flattop = (2 * np.abs(f_flattop_fft[indice_pico_flattop, np.arange(R)])) / np.mean(sg.flattop(N))


# Estimador de frecuencia
om_flattop = ff_vector_pos[indice_pico_flattop]


####################################################
#BLACKMANHARRIS
####################################################

# Genero una funcion que sea la multiplicada por mi ventana
f_blackman =  f_sen * sg.blackmanharris(N).reshape(N, 1)
#Por que se le pone el reshape a este

f_blackman_fft = (1/N) * np.fft.fft(f_blackman, axis=0)

# Formula de potencia --> elevo al cuadrado y divido por 2 para tener Watts
potencia_lineal_blackman = (np.abs(f_blackman_fft[:N // 2, :]))**2
 
dens_pot_blackman = 10 * np.log10(potencia_lineal_blackman + 1e-12)

# Estimador de amplitud
indice_pico_blackman = np.argmax(np.abs(f_blackman_fft[:N // 2, :]), axis=0)

a_1_2_blackman = (2 * np.abs(f_blackman_fft[indice_pico_blackman, np.arange(R)])) / np.mean(sg.blackmanharris(N))

# Estimador de frecuencia
om_blackman = ff_vector_pos[indice_pico_blackman]


####################################################
#HANN
####################################################

# Genero una funcion que sea la multiplicada por mi ventana
f_hann =  f_sen * sg.hann(N).reshape(N, 1)
#Por que se le pone el reshape a este

f_hann_fft = (1/N) * np.fft.fft(f_hann, axis=0)

# Formula de potencia --> elevo al cuadrado y divido por 2 para tener Watts
potencia_lineal_hann = (np.abs(f_hann_fft[:N // 2, :]))**2
 
dens_pot_hann = 10 * np.log10(potencia_lineal_hann + 1e-12)


# Estimador de amplitud
indice_pico_hann = np.argmax(np.abs(f_hann_fft[:N // 2, :]), axis=0)

a_1_2_hann = (2 * np.abs(f_hann_fft[indice_pico_hann, np.arange(R)])) / np.mean(sg.hann(N))


# Estimador de frecuencia
om_hann = ff_vector_pos[indice_pico_hann]


#%% Graficos

######################################
# Grafico de ventana rectangular
######################################

plt.figure()
plt.plot(ff_vector_pos, dens_pot, alpha=0.3)
plt.title('Módulo Espectral de las 200 Realizaciones (Matriz 1000 x 200)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Decibeles')
plt.xlim(240, 260) 
plt.grid(True)
plt.show()

######################################
# Grafico de ventana flattop
######################################

plt.figure()
plt.plot(ff_vector_pos, dens_pot_flattop, alpha=0.3)
plt.title('Módulo Espectral de las 200 Realizaciones (Matriz 1000 x 200)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Decibeles')
plt.xlim(240, 260) 
plt.grid(True)
plt.show()


######################################
# Grafico de ventana blackman
######################################

plt.figure()
plt.plot(ff_vector_pos, dens_pot_blackman, alpha=0.3)
plt.title('Módulo Espectral de las 200 Realizaciones (Matriz 1000 x 200)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Decibeles')
plt.xlim(240, 260) 
plt.grid(True)
plt.show()

######################################
# Grafico de ventana blackman
######################################

plt.figure()
plt.plot(ff_vector_pos, dens_pot_hann, alpha=0.3)
plt.title('Módulo Espectral de las 200 Realizaciones (Matriz 1000 x 200)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Decibeles')
plt.xlim(240, 260) 
plt.grid(True)
plt.show()

######################################
# Grafico de los histogramas
######################################

# FRECUENCIA

plt.figure()
plt.hist(om_blackman, bins = 20, alpha = 0.3, label= 'Blackman', color= 'Green')
plt.hist(om_f, bins = 20, alpha = 0.3, label= 'Rectangular', color= 'Pink')
plt.hist(om_flattop, bins = 20, alpha = 0.3, label= 'Flattop', color= 'Red')
plt.hist(om_hann, bins = 20, alpha = 0.3, label= 'Hann', color= 'Blue')
plt.ylim(0, 80)
plt.legend(['Blackman = Verde', 'Rectangular = Rosa', 'Flattop = Rojo', 'Hann = Azul'])
plt.xlabel('Frecuencia estimada')
plt.ylabel('Cantidad de realizaciones')
plt.title('Histograma de estimadores de frecuencia 10dB')
plt.show() 

# AMPLITUD

plt.figure()
plt.hist(a_1_2_blackman, bins = 20, alpha = 1, label= 'Blackman', color= 'Green')
plt.hist(a_1_2, bins = 20, alpha = 0.8, label= 'Rectangular', color= 'Pink')
plt.hist(a_1_2_flattop, bins = 20, alpha = 0.6, label= 'Flattop', color= 'Red')
plt.hist(a_1_2_hann, bins = 20, alpha = 0.3, label= 'Hann', color= 'Blue')
plt.legend(['Blackman = Verde', 'Rectangular = Rosa', 'Flattop = Rojo', 'Hann = Azul'])
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.title('Histograma de estimadores de amplitud 10dB')
plt.show() 

#%% Cuadros de estimadores



###########################
# Sesgo rectangular
###########################

MediaRec = np.mean(a_1_2) #Calcula el valor medio de las estimaciones que hace
VarRec = np.var(a_1_2)
# Calculo sesgo, restando el valor original con obtenido

sesgoRec = MediaRec - a0

print('ESTIMADORES DE AMPLITUD')

print('---------------------------------------------')
print('Sesgo y Varianza de ventana rectangular en 10dB')
print('Sesgo rec= ', sesgoRec)
print('Varianza rec=', VarRec)

###########################
# Sesgo flattop
###########################

MediaFlat = np.mean(a_1_2_flattop)
VarFlat = np.var(a_1_2_flattop)
sesgoFlat = MediaFlat - a0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana falttop en 10dB')
print('Sesgo flat= ', sesgoFlat)
print('Varianza flat=', VarFlat)

###########################
# Sesgo blackman
###########################

MediaBlack = np.mean(a_1_2_blackman)
VarBlack = np.var(a_1_2_blackman)
sesgoBlack = MediaBlack - a0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana blackman en 10dB')
print('Sesgo black= ', sesgoBlack)
print('Varianza black=', VarBlack)

###########################
# Sesgo Hann
###########################

MediaHann = np.mean(a_1_2_hann)
VarHann = np.var(a_1_2_hann)
sesgoHann = MediaHann - a0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana Hann en 10dB')
print('Sesgo black= ', sesgoHann)
print('Varianza black=', VarHann)

##################################################################################
#FRECUENCIA
##################################################################################

###########################
# Sesgo rectangular
###########################

MediaRecOm = np.mean(om_f) 
VarRecOm = np.var(om_f)


sesgoRecOm = MediaRecOm - omega0

print('________________________________________________ ')
print('ESTIMADORES DE FRECUENCIA')

print('---------------------------------------------')
print('Sesgo y Varianza de ventana rectangular en 10dB')
print('Sesgo rec= ', sesgoRecOm)
print('Varianza rec=', VarRecOm)

###########################
# Sesgo flattop
###########################

MediaFlatOm = np.mean(om_flattop)
VarFlatOm = np.var(om_flattop)
sesgoFlatOm = MediaFlatOm - omega0 

print('---------------------------------------------')
print('Sesgo y Varianza de ventana falttop en 10dB')
print('Sesgo flat= ', sesgoFlatOm)
print('Varianza flat=', VarFlatOm)

###########################
# Sesgo blackman
###########################

MediaBlackOm = np.mean(om_blackman)
VarBlackOm = np.var(om_blackman)
sesgoBlackOm = MediaBlackOm - omega0 

print('---------------------------------------------')
print('Sesgo y Varianza de ventana blackman en 10dB')
print('Sesgo black= ', sesgoBlackOm)
print('Varianza black=', VarBlackOm)

###########################
# Sesgo Hann
###########################

MediaHannOm = np.mean(om_hann)
VarHannOm = np.var(om_hann)
sesgoHannOm = MediaHannOm - omega0 

print('---------------------------------------------')
print('Sesgo y Varianza de ventana Hann en 10dB')
print('Sesgo black= ', sesgoHannOm)
print('Varianza black=', VarHannOm)

#%% EXPERIMENTO 3 dB

snr_db_3 = 3
varianza_3 = P_s / (10**(snr_db_3 / 10))
sigma_3 = np.sqrt(varianza_3)

#%% Condicionando las variables

# Hacer las de ruido
Na_3 = np.random.normal(0.0, sigma_3, (N, R))

fr_3 = np.random.uniform(-limiteUniforme, limiteUniforme, R)

# Matrices
n_col_3 = np.arange(N).reshape(N, 1)

omega1_3 = 2 * np.pi * (omega0 + fr_3) / fs

omega1_3 = omega1_3.reshape(1, R)

#%% Creacion de funciones

# Hago mi funcion
f_sen_3 = a0 * np.sin(omega1_3 * n_col_3) + Na_3


#FFT
f_fft_3 = (1/N) * np.fft.fft(f_sen_3, axis=0)

potencia_lineal_3 = (np.abs(f_fft_3[:N // 2, :]))**2

dens_pot_3 = 10 * np.log10(potencia_lineal_3 + 1e-12)


# Estimador de frecuencia
indice_pico_3 = np.argmax(np.abs(f_fft_3[:N // 2, :]), axis=0)

om_f_3 = ff_vector_pos[indice_pico_3]

# Estimador de amplitud
a_1_2_3 = 2 * np.abs(f_fft_3[indice_pico_3, np.arange(R)])


#%% Ventanas

# La rectangular ya esta hecha porque es multiplicado por uno


####################################################
# FLATTOP
####################################################

f_flattop_3 = f_sen_3 * sg.flattop(N).reshape(N, 1)

f_flattop_fft_3 = (1/N) * np.fft.fft(f_flattop_3, axis=0)

potencia_lineal_flattop_3 = (np.abs(f_flattop_fft_3[:N // 2, :]))**2

dens_pot_flattop_3 = 10 * np.log10(potencia_lineal_flattop_3 + 1e-12)

# Estimador de amplitud
indice_pico_flattop_3 = np.argmax(np.abs(f_flattop_fft_3[:N // 2, :]), axis=0)

a_1_2_flattop_3 = (2 * np.abs(f_flattop_fft_3[indice_pico_flattop_3, np.arange(R)])) / np.mean(sg.flattop(N))

# Estimador de frecuencia
om_flattop_3 = ff_vector_pos[indice_pico_flattop_3]


####################################################
# BLACKMANHARRIS
####################################################

f_blackman_3 = f_sen_3 * sg.blackmanharris(N).reshape(N, 1)

f_blackman_fft_3 = (1/N) * np.fft.fft(f_blackman_3, axis=0)

potencia_lineal_blackman_3 = (np.abs(f_blackman_fft_3[:N // 2, :]))**2

dens_pot_blackman_3 = 10 * np.log10(potencia_lineal_blackman_3 + 1e-12)

# Estimador de amplitud
indice_pico_blackman_3 = np.argmax(np.abs(f_blackman_fft_3[:N // 2, :]), axis=0)

a_1_2_blackman_3 = (2 * np.abs(f_blackman_fft_3[indice_pico_blackman_3, np.arange(R)])) / np.mean(sg.blackmanharris(N))

# Estimador de frecuencia
om_blackman_3 = ff_vector_pos[indice_pico_blackman_3]


####################################################
# HANN
####################################################

f_hann_3 = f_sen_3 * sg.hann(N).reshape(N, 1)

f_hann_fft_3 = (1/N) * np.fft.fft(f_hann_3, axis=0)

potencia_lineal_hann_3 = (np.abs(f_hann_fft_3[:N // 2, :]))**2

dens_pot_hann_3 = 10 * np.log10(potencia_lineal_hann_3 + 1e-12)

# Estimador de amplitud
indice_pico_hann_3 = np.argmax(np.abs(f_hann_fft_3[:N // 2, :]), axis=0)

a_1_2_hann_3 = (2 * np.abs(f_hann_fft_3[indice_pico_hann_3, np.arange(R)])) / np.mean(sg.hann(N))

# Estimador de frecuencia
om_hann_3 = ff_vector_pos[indice_pico_hann_3]


#%% Graficos

######################################
# Grafico de ventana rectangular
######################################

plt.figure()
plt.plot(ff_vector_pos, dens_pot_3, alpha=0.3)
plt.title('Módulo Espectral de las 200 Realizaciones - 3dB')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Decibeles')
plt.xlim(240, 260)
plt.grid(True)
plt.show()


######################################
# Grafico de ventana flattop
######################################

plt.figure()
plt.plot(ff_vector_pos, dens_pot_flattop_3, alpha=0.3)
plt.title('Módulo Espectral de las 200 Realizaciones - Flattop - 3dB')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Decibeles')
plt.xlim(240, 260)
plt.grid(True)
plt.show()


######################################
# Grafico de ventana blackman
######################################

plt.figure()
plt.plot(ff_vector_pos, dens_pot_blackman_3, alpha=0.3)
plt.title('Módulo Espectral de las 200 Realizaciones - Blackman-Harris - 3dB')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Decibeles')
plt.xlim(240, 260)
plt.grid(True)
plt.show()


######################################
# Grafico de ventana Hann
######################################

plt.figure()
plt.plot(ff_vector_pos, dens_pot_hann_3, alpha=0.3)
plt.title('Módulo Espectral de las 200 Realizaciones - Hann - 3dB')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Decibeles')
plt.xlim(240, 260)
plt.grid(True)
plt.show()


######################################
# Grafico de los histogramas
######################################

# FRECUENCIA

plt.figure()
plt.hist(om_blackman_3, bins=20, alpha=0.3, label='Blackman', color='Green')
plt.hist(om_f_3, bins=20, alpha=0.3, label='Rectangular', color='Pink')
plt.hist(om_flattop_3, bins=20, alpha=0.3, label='Flattop', color='Red')
plt.hist(om_hann_3, bins=20, alpha=0.3, label='Hann', color='Blue')
plt.ylim(0, 80)
plt.legend(['Blackman = Verde', 'Rectangular = Rosa', 'Flattop = Rojo', 'Hann = Azul'])
plt.xlabel('Frecuencia estimada [Hz]')
plt.ylabel('Cantidad de realizaciones')
plt.title('Histograma de estimadores de frecuencia 3dB')
plt.show()


# AMPLITUD

plt.figure()
plt.hist(a_1_2_blackman_3, bins=20, alpha=1, label='Blackman', color='Green')
plt.hist(a_1_2_3, bins=20, alpha=0.8, label='Rectangular', color='Pink')
plt.hist(a_1_2_flattop_3, bins=20, alpha=0.6, label='Flattop', color='Red')
plt.hist(a_1_2_hann_3, bins=20, alpha=0.3, label='Hann', color='Blue')
plt.legend(['Blackman = Verde', 'Rectangular = Rosa', 'Flattop = Rojo', 'Hann = Azul'])
plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')
plt.title('Histograma de estimadores de amplitud 3dB')
plt.show()


#%% Cuadros de estimadores 3dB


###########################
# AMPLITUD
###########################

MediaRec_3 = np.mean(a_1_2_3)
VarRec_3 = np.var(a_1_2_3)

sesgoRec_3 = MediaRec_3 - a0

print('ESTIMADORES DE AMPLITUD - 3dB')

print('---------------------------------------------')
print('Sesgo y Varianza de ventana rectangular en 3dB')
print('Sesgo rec= ', sesgoRec_3)
print('Varianza rec=', VarRec_3)


###########################
# Sesgo flattop
###########################

MediaFlat_3 = np.mean(a_1_2_flattop_3)
VarFlat_3 = np.var(a_1_2_flattop_3)

sesgoFlat_3 = MediaFlat_3 - a0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana flattop en 3dB')
print('Sesgo flat= ', sesgoFlat_3)
print('Varianza flat=', VarFlat_3)


###########################
# Sesgo blackman
###########################

MediaBlack_3 = np.mean(a_1_2_blackman_3)
VarBlack_3 = np.var(a_1_2_blackman_3)

sesgoBlack_3 = MediaBlack_3 - a0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana blackman en 3dB')
print('Sesgo black= ', sesgoBlack_3)
print('Varianza black=', VarBlack_3)


###########################
# Sesgo Hann
###########################

MediaHann_3 = np.mean(a_1_2_hann_3)
VarHann_3 = np.var(a_1_2_hann_3)

sesgoHann_3 = MediaHann_3 - a0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana Hann en 3dB')
print('Sesgo Hann= ', sesgoHann_3)
print('Varianza Hann=', VarHann_3)


##################################################################################
# FRECUENCIA
##################################################################################

###########################
# Sesgo rectangular
###########################

MediaRecOm_3 = np.mean(om_f_3)
VarRecOm_3 = np.var(om_f_3)

sesgoRecOm_3 = MediaRecOm_3 - omega0

print('________________________________________________ ')
print('ESTIMADORES DE FRECUENCIA - 3dB')

print('---------------------------------------------')
print('Sesgo y Varianza de ventana rectangular en 3dB')
print('Sesgo rec= ', sesgoRecOm_3)
print('Varianza rec=', VarRecOm_3)


###########################
# Sesgo flattop
###########################

MediaFlatOm_3 = np.mean(om_flattop_3)
VarFlatOm_3 = np.var(om_flattop_3)

sesgoFlatOm_3 = MediaFlatOm_3 - omega0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana flattop en 3dB')
print('Sesgo flat= ', sesgoFlatOm_3)
print('Varianza flat=', VarFlatOm_3)


###########################
# Sesgo blackman
###########################

MediaBlackOm_3 = np.mean(om_blackman_3)
VarBlackOm_3 = np.var(om_blackman_3)

sesgoBlackOm_3 = MediaBlackOm_3 - omega0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana blackman en 3dB')
print('Sesgo black= ', sesgoBlackOm_3)
print('Varianza black=', VarBlackOm_3)


###########################
# Sesgo Hann
###########################

MediaHannOm_3 = np.mean(om_hann_3)
VarHannOm_3 = np.var(om_hann_3)

sesgoHannOm_3 = MediaHannOm_3 - omega0

print('---------------------------------------------')
print('Sesgo y Varianza de ventana Hann en 3dB')
print('Sesgo Hann= ', sesgoHannOm_3)
print('Varianza Hann=', VarHannOm_3)
