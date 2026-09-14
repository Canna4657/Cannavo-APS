# -*- coding: utf-8 -*-
"""
Análisis y Procesamiento de Señales (APS)
Tarea Semanal 3: Desparramo Espectral, Teorema de Parseval y Zero Padding
Autores: Pedro Joaquin Cannavo y equipo
"""

import numpy as np
import matplotlib.pyplot as plt

#####################################################################
# PARÁMETROS GENERALES
#####################################################################

N = 1000                  # Cantidad de muestras originales
fs = 1000.0               # Frecuencia de muestreo [Hz]
df = fs / N               # Resolución espectral original: 1 Hz

R = 1.0                   # Resistencia normalizada [ohm]
vmax = np.sqrt(2)         # Amplitud pico para lograr potencia unitaria (1 W)
dc = 0.0
ph = 0.0

# Eje temporal original (0 a 1 segundo)
tt = np.arange(N) / fs

def mi_funcion_sen(vmax=np.sqrt(2), dc=0, ff=1, ph=0, nn=N, fs=fs):
    t = np.arange(nn) / fs
    x = vmax * np.sin(2 * np.pi * ff * t + ph) + dc
    return t, x


#####################################################################
# DEFINICIÓN DE LAS TRES SENOIDALES (INCISO A)
#####################################################################

# Frecuencias solicitadas
f1 = (N / 4) * df          # 250.0 Hz (k0 = 250 - Coherente)
f2 = (N / 4 + 0.25) * df   # 250.25 Hz (k0 = 250.25 - Desintonía leve)
f3 = (N / 4 + 0.5) * df    # 250.5 Hz (k0 = 250.5 - Desintonía máxima)

tt, x1 = mi_funcion_sen(vmax=vmax, dc=dc, ff=f1, ph=ph, nn=N, fs=fs)
tt, x2 = mi_funcion_sen(vmax=vmax, dc=dc, ff=f2, ph=ph, nn=N, fs=fs)
tt, x3 = mi_funcion_sen(vmax=vmax, dc=dc, ff=f3, ph=ph, nn=N, fs=fs)

# Eje de frecuencias original (0 a fs/2 = 500 Hz)
ff_pos = np.fft.rfftfreq(N, 1/fs)

# FFT unilateral normalizada
X1 = np.fft.rfft(x1) / N
X2 = np.fft.rfft(x2) / N
X3 = np.fft.rfft(x3) / N

# Densidad Espectral de Potencia en dB (0 dB = 1 W)
X1_dB = 10 * np.log10(2 * np.abs(X1)**2 + 1e-15)
X2_dB = 10 * np.log10(2 * np.abs(X2)**2 + 1e-15)
X3_dB = 10 * np.log10(2 * np.abs(X3)**2 + 1e-15)


#####################################################################
# GRÁFICO 1: INCISO A - PSD ORIGINAL
#####################################################################

plt.figure(1, figsize=(11, 5))
plt.plot(ff_pos, X1_dB, 'o:', label=f'k = N/4 ({f1:.2f} Hz) - Coherente', color='#FF007F', lw=1)
plt.plot(ff_pos, X2_dB, 'o:', label=f'k = N/4 + 0.25 ({f2:.2f} Hz)', color='#00A8FF', lw=1)
plt.plot(ff_pos, X3_dB, 'o:', label=f'k = N/4 + 0.5 ({f3:.2f} Hz) - Máx. desintonía', color='#00FF80', lw=1)

plt.title('Inciso a: Densidad Espectral de Potencia (PSD) - Desparramo Espectral')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlim(230, 270)       # Zoom en la zona de desparramo
plt.ylim(-60, 5)
plt.grid(True, alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()


#####################################################################
# INCISO B - VERIFICACIÓN DE POTENCIA MEDIANTE PARSEVAL
#####################################################################
# 1. Potencia en el tiempo (promedio de las muestras al cuadrado)
P1_tiempo = np.mean(x1**2)
P2_tiempo = np.mean(x2**2)
P3_tiempo = np.mean(x3**2)
# 2. Potencia en frecuencia usando Parseval: sum(|X|^2) / N^2
P1_freq = np.sum(np.abs(np.fft.fft(x1))**2) / (N**2)
P2_freq = np.sum(np.abs(np.fft.fft(x2))**2) / (N**2)
P3_freq = np.sum(np.abs(np.fft.fft(x3))**2) / (N**2)
print("\n=================================================")
print("INCISO B: COMPROBACIÓN DEL TEOREMA DE PARSEVAL")
print("=================================================")
print("Potencia calculada en el tiempo:")
print(f"  Senoidal 1 (250.00 Hz): {P1_tiempo:.6f} W")
print(f"  Senoidal 2 (250.25 Hz): {P2_tiempo:.6f} W")
print(f"  Senoidal 3 (250.50 Hz): {P3_tiempo:.6f} W")
print("\nPotencia calculada en frecuencia (Parseval):")
print(f"  Senoidal 1 (250.00 Hz): {P1_freq:.6f} W")
print(f"  Senoidal 2 (250.25 Hz): {P2_freq:.6f} W")
print(f"  Senoidal 3 (250.50 Hz): {P3_freq:.6f} W")
print("-------------------------------------------------")
print("Conclusión: En los tres casos la potencia se conserva en 1 W.")
print("La caída del pico se debe a que la energía se desparramó a los bines vecinos.")
print("=================================================\n")


#####################################################################
# INCISO C - ZERO PADDING (9*N CEROS AL FINAL)
#####################################################################

# Agregamos 9*N ceros al final de cada senal (largo total = 10*N = 10000 muestras)
# Matematicamente hablando el zeropadding no te afecta la FFT pq en la sumatoria se multiplica por el x(n)
# Y como vos agregas cero entonces no le afectas el resultado
# LO qu ecambia es la frecuencia en la que le pedis qu ete de los resultados, antes siendo cada 1Hz y ahora 0.1
# Lo que si cmabia es el pormedio de potencia en el tiempo pq tenes 9 segundfos de silencio.

cant_ceros = 9 * N
N_zp = N + cant_ceros     # los Ns mas el zeropadding
df_zp = fs / N_zp         # Nueva grilla de frecuencias: df_zp = 0.1 Hz

# Padding con numpy
x1_zp = np.pad(x1, (0, cant_ceros), mode='constant')
x2_zp = np.pad(x2, (0, cant_ceros), mode='constant')
x3_zp = np.pad(x3, (0, cant_ceros), mode='constant')

# Eje de frecuencias con Zero Padding (0 a 500 Hz con paso de 0.1 Hz)
ff_zp = np.fft.rfftfreq(N_zp, 1/fs)

# Normalizamos por N (la cantidad de muestras de señal activa) para mantener 0 dB como 1 W
X1_zp = np.fft.rfft(x1_zp) / N
X2_zp = np.fft.rfft(x2_zp) / N
X3_zp = np.fft.rfft(x3_zp) / N

X1_zp_dB = 10 * np.log10(2 * np.abs(X1_zp)**2 + 1e-15)
X2_zp_dB = 10 * np.log10(2 * np.abs(X2_zp)**2 + 1e-15)
X3_zp_dB = 10 * np.log10(2 * np.abs(X3_zp)**2 + 1e-15)


#####################################################################
# GRÁFICO 2: INCISO C - COMPARACIÓN ZERO PADDING VS ORIGINAL
#####################################################################

fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# Subplot 1: k = N/4
axs[0].plot(ff_zp, X1_zp_dB, color='#FF007F', label='Con Zero Padding (10*N) - Continuo', lw=1.2)
axs[0].plot(ff_pos, X1_dB, 'ko', markersize=4, label='Sin Zero Padding (N) - Muestras')
axs[0].set_title(f'k = N/4 (f0 = {f1:.2f} Hz) - Coherente')
axs[0].set_ylabel('Potencia [dB]')
axs[0].set_ylim(-60, 5)
axs[0].grid(True, alpha=0.4)
axs[0].legend(loc='upper right', fontsize=9)

# Subplot 2: k = N/4 + 0.25
axs[1].plot(ff_zp, X2_zp_dB, color='#00A8FF', label='Con Zero Padding (10*N) - Continuo', lw=1.2)
axs[1].plot(ff_pos, X2_dB, 'ko', markersize=4, label='Sin Zero Padding (N) - Muestras')
axs[1].set_title(f'k = N/4 + 0.25 (f0 = {f2:.2f} Hz) - Desintonía intermedia')
axs[1].set_ylabel('Potencia [dB]')
axs[1].set_ylim(-60, 5)
axs[1].grid(True, alpha=0.4)
axs[1].legend(loc='upper right', fontsize=9)

# Subplot 3: k = N/4 + 0.5
axs[2].plot(ff_zp, X3_zp_dB, color='#00FF80', label='Con Zero Padding (10*N) - Continuo', lw=1.2)
axs[2].plot(ff_pos, X3_dB, 'ko', markersize=4, label='Sin Zero Padding (N) - Muestras')
axs[2].set_title(f'k = N/4 + 0.5 (f0 = {f3:.2f} Hz) - Máxima desintonía')
axs[2].set_xlabel('Frecuencia [Hz]')
axs[2].set_ylabel('Potencia [dB]')
axs[2].set_xlim(235, 265)   # Zoom alrededor de 250 Hz
axs[2].set_ylim(-60, 5)
axs[2].grid(True, alpha=0.4)
axs[2].legend(loc='upper right', fontsize=9)

fig.suptitle('Inciso c: Efecto de Zero Padding (Interpolación de la DTFT)', fontsize=13, y=0.99)
plt.tight_layout()

# Mostrar todas las figuras
plt.show()
