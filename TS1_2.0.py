import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

####################FUNCIONES##############################################

def funcion_seno(vmax, dc, ff, ph, nn, fs):
    
    tt = (np.arange(nn)/fs).reshape(-1, 1)
    
    xx = (vmax * np.sin(2 * np.pi * ff * tt + ph) + dc).reshape(-1, 1)
    
    return tt, xx

###########################################################################



#Para cada senal ver su modulo de la transformada
##################################################################################################
#Hago senal sinuoidal de 2000hz con muestreo de mas de 10 puntos por ciclo por ende minimo 20k

#Cantidad de muestras
N = 1000

#Frecuencia de muestreo 
fs = 25000

#Resistencia asumida
R = 1


##################################################################################################
#SEÑAL DE REFERENCIA PARA CALIBRAR EL ESPECTRO
##################################################################################################

#El profesor pide tomar una senoidal de 1 W como referencia de 0 dB

P_ref = 1

#Para R=1 ohm:
#P = Vrms^2 / R
Vrms_ref = np.sqrt(P_ref * R)

#Para una senoidal:
#Vrms = Vmax / sqrt(2)
vmax_ref = Vrms_ref * np.sqrt(2)

#Creo la senal de referencia de 1 W
tt_ref, xx_ref = funcion_seno(vmax=vmax_ref, dc=0, ff=2000,
                              ph=0, nn=N, fs=fs)

#FFT de la senal de referencia
xx_ref_fft = np.fft.fft(xx_ref.flatten())

#Modulo del espectro de referencia
xx_ref_abs = np.abs(xx_ref_fft)

#Tomo el maximo del espectro como referencia para 0 dB
espectro_ref = np.max(xx_ref_abs)

print('Valor maximo del espectro de referencia:', espectro_ref)

##################################################################################################



#PRIMER SINUSOIDAL
##################################################################################################

#Primer sinuoidal
tt1, xx1 = funcion_seno(vmax=1, dc=0, ff=2000, ph=0, nn=N, fs=fs)

#Potencia media de la primera senal
P1 = np.mean(xx1**2) / R
print('Potencia media senal 1 =', P1, 'W')

#FFT de la funcion para tenerlo representado en frecuencias
xx1_fft = np.fft.fft(xx1.flatten())

#Defino el eje en frecuencia para poder graficar
ff_eje = np.fft.fftfreq(N, 1/fs)

#Defino el modulo
xx1_abs = np.abs(xx1_fft)

#Normalizo usando el espectro de referencia
xx1_norm = xx1_abs / espectro_ref

#Paso a decibeles
xx1_db = 20 * np.log10(xx1_norm + 1e-12)

#Reordeno eje y espectro con fftshift para que quede continuo de -fs/2 a fs/2
ff_eje_shift = np.fft.fftshift(ff_eje)
xx1_db_shift = np.fft.fftshift(xx1_db)

#Grafico la funcion
plt.figure(1)
plt.plot(ff_eje_shift, xx1_db_shift, color='red')
plt.title('FFT modulo 1')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.show()



##################################################################################################
#SEGUNDA SINUSOIDAL
##################################################################################################

#Aca hago la segunda senoidal que me piden que es con 2 W de potencia media
#y desfase de pi/2

#Calculo el dato de potencia media
Pmed = 2

Vrms = np.sqrt(Pmed * R)

vmax = Vrms * np.sqrt(2)

print('Amplitud segunda senal:', vmax)

#Creo la funcion
tt2, xx2 = funcion_seno(vmax, dc=0, ff=2000, ph=np.pi/2,
                        nn=N, fs=fs)

#FFT de la funcion para tenerlo representado en frecuencias
xx2_fft = np.fft.fft(xx2.flatten())

#Defino el modulo
xx2_abs = np.abs(xx2_fft)

#Normalizo usando el espectro de referencia
xx2_norm = xx2_abs / espectro_ref

#Paso a decibeles
xx2_db = 20 * np.log10(xx2_norm + 1e-12)

#Reordeno eje y espectro con fftshift
xx2_db_shift = np.fft.fftshift(xx2_db)

#Grafico la funcion
plt.figure(2)
plt.plot(ff_eje_shift, xx2_db_shift, color='green')
plt.title('FFT modulo 2')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.show()



###################################################################################################
#RUIDO NORMALMENTE DISTRIBUIDO
###################################################################################################

#Una secuencia aleatoria de ruido normalmente distribuido
#con DC (valor medio) 0V y varianza 0.1 W.

varianza1 = 0.1

sigma = np.sqrt(varianza1)

#Ahora que tengo esto puedo hacer mi ruido
#Creo ruido y luego armo mi funcion completa con ruido
ruido1 = np.random.normal(0, sigma, N).reshape(-1, 1)

xx1_ruidosa = xx1 + ruido1

#FFT de la funcion para tenerlo representado en frecuencias
xx1R_fft = np.fft.fft(xx1_ruidosa.flatten())

#Defino el modulo
xx1R_abs = np.abs(xx1R_fft)

#Normalizo usando el espectro de referencia
xx1R_norm = xx1R_abs / espectro_ref

#Paso a decibeles
xx1R_db = 20 * np.log10(xx1R_norm + 1e-12)

#Reordeno eje y espectro con fftshift
xx1R_db_shift = np.fft.fftshift(xx1R_db)

#Grafico la funcion
plt.figure(3)
plt.plot(ff_eje_shift, xx1R_db_shift, color='pink')
plt.title('FFT modulo de ruido normal')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.show()



###################################################################################################
#RUIDO UNIFORMEMENTE DISTRIBUIDO
###################################################################################################

#Una secuencia aleatoria de ruido uniformemente distribuido
#con DC (valor medio) 0V y varianza 0.1 W. 

#Con ruido uniforme es un toque distinto pq lo calculo distinto,
#tengo que poner un intervalo

varianza2 = 0.1

#Varianza para intervalos simetricos es igual a a al cuadrado sobre 3
#para este tipo de distribucion
#despejo para intervalo de [-a:a]
a = np.sqrt(0.1 * 3)

print('a es:', a)

#Creo ruido y luego armo mi funcion completa con ruido
ruido2 = np.random.uniform(-a, a, N).reshape(-1, 1)

xx2_ruidosa = xx1 + ruido2

#FFT de la funcion para tenerlo representado en frecuencias
xx2R_fft = np.fft.fft(xx2_ruidosa.flatten())

#Defino el modulo
xx2R_abs = np.abs(xx2R_fft)

#Normalizo usando el espectro de referencia
xx2R_norm = xx2R_abs / espectro_ref

#Paso a decibeles
xx2R_db = 20 * np.log10(xx2R_norm + 1e-12)

#Reordeno eje y espectro con fftshift
xx2R_db_shift = np.fft.fftshift(xx2R_db)

#Grafico la funcion
plt.figure(4)
plt.plot(ff_eje_shift, xx2R_db_shift, color='violet')
plt.title('FFT modulo de ruido uniforme')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.show()



###################################################################################################
#PULSO RECTANGULAR
###################################################################################################

#Un pulso rectangular de la misma frecuencia, 1 W de potencia
#y ciclo de actividad del 50%.

ff = 2000

pulso = signal.square(2*np.pi*ff*tt1).reshape(-1, 1)

#Potencia instantanea es igual a Vmax al cuadrado,
#en este caso Vmax es uno y menos uno que cumplen
#Tambien se puede justificar que potencia media es Vrms al cuadrado
#y como es simetrica Vrms es igual a Vmax

pot = np.mean(pulso**2) / R

print('Potencia media pulso =', pot, 'W')

#FFT de la funcion para tenerlo representado en frecuencias
xx3_fft = np.fft.fft(pulso.flatten())

#Defino el modulo
xx3_abs = np.abs(xx3_fft)

#Normalizo usando el espectro de referencia
xx3_norm = xx3_abs / espectro_ref

#Paso a decibeles
xx3_db = 20 * np.log10(xx3_norm + 1e-12)

#Reordeno eje y espectro con fftshift
xx3_db_shift = np.fft.fftshift(xx3_db)

#Grafico la funcion
plt.figure(5)
plt.plot(ff_eje_shift, xx3_db_shift, color='violet') 
plt.title('FFT pulso rectangular')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.show()