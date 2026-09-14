import numpy as np
import matplotlib.pyplot as plt

#####################################################################
# SECCION 1: PARAMETROS GENERALES Y SENAL SENOIDAL
#####################################################################

N = 1000                  # Cantidad de muestras
fs = 1000.0               # Frecuencia de muestreo [Hz]
VF = 2.0                  # Rango analogico (+/- VR o +/- VF) [V]
R = 1.0                   # Resistencia normalizada [ohm]

# Frecuencia centrada exactamente en un bin (delta f = fs / N = 1 Hz)
f0 = fs / N
tt = np.arange(N) / fs

# Senoidal con varianza / potencia unitaria (Ps = 1 W)
s = np.sqrt(2) * np.sin(2 * np.pi * f0 * tt)
Ps = np.mean(s**2) / R

# Eje de frecuencias unilaterales (0 a fs/2 = 500 Hz)
ff = np.fft.rfftfreq(N, 1/fs)
S_orig_db = 10 * np.log10(2 * np.abs(np.fft.rfft(s))**2 / (N**2) + 1e-15)


#####################################################################
# FUNCION AUXILIAR: SIMULACION DE UN ADC
#####################################################################

def procesar_adc(B, kn):
    """
    Calcula la digitalizacion, espectros y pisos de ruido para B bits y kn.
    """
    q = VF / (2**B)
    Pq_teorica = (q**2) / 12.0
    Pn_teorica = kn * Pq_teorica
    
    # Ruido analogico gaussiano
    sigma_n = np.sqrt(Pn_teorica)
    ruido_analogico = np.random.normal(0, sigma_n, N)
    sR = s + ruido_analogico
    
    # Cuantizacion uniforme con saturacion
    sR_clip = np.clip(sR, -VF, VF)
    s_cuantizada = np.round(sR_clip / q) * q
    ruido_cuantizacion = s_cuantizada - sR_clip
    
    # Potencias medidas
    Pn_medida = np.mean(ruido_analogico**2)
    Pq_medida = np.mean(ruido_cuantizacion**2)
    
    SNR_antes = 10 * np.log10(Ps / Pn_medida)
    SNR_despues = 10 * np.log10(Ps / (Pn_medida + Pq_medida))
    
    # Espectros
    S_in_db  = 10 * np.log10(2 * np.abs(np.fft.rfft(sR))**2 / (N**2) + 1e-15)
    S_out_db = 10 * np.log10(2 * np.abs(np.fft.rfft(s_cuantizada))**2 / (N**2) + 1e-15)
    
    # Pisos teoricos
    piso_analog = 10 * np.log10(Pn_teorica / (N / 2))
    piso_digital = 10 * np.log10(Pq_teorica / (N / 2))
    
    return {
        'B': B, 'kn': kn, 'q': q,
        'sR': sR, 's_cuantizada': s_cuantizada, 'ruido_cuantizacion': ruido_cuantizacion,
        'Pq': Pq_teorica, 'Pn': Pn_teorica,
        'SNR_antes': SNR_antes, 'SNR_despues': SNR_despues,
        'S_in_db': S_in_db, 'S_out_db': S_out_db,
        'piso_analog': piso_analog, 'piso_digital': piso_digital
    }


def graficar_pisos(ax, piso_analog, piso_digital):
    """
    Grafica los pisos de ruido evitando que una linea tape completamente a la otra
    cuando coinciden (caso kn = 1).
    """
    if np.isclose(piso_analog, piso_digital, atol=0.2):
        # Guiones alternados (rojo y cyan sobre el mismo nivel)
        ax.axhline(piso_analog, color='red', linestyle=(0, (4, 4)), lw=1.5,
                   label=f'$\\bar{{n}} = {piso_analog:.1f}$ dB (piso analog.)')
        ax.axhline(piso_digital, color='cyan', linestyle=(4, (4, 4)), lw=1.5,
                   label=f'$\\bar{{n_Q}} = {piso_digital:.1f}$ dB (piso digital)')
    else:
        ax.axhline(piso_analog, color='red', linestyle='--', lw=1.3,
                   label=f'$\\bar{{n}} = {piso_analog:.1f}$ dB (piso analog.)')
        ax.axhline(piso_digital, color='cyan', linestyle='--', lw=1.3,
                   label=f'$\\bar{{n_Q}} = {piso_digital:.1f}$ dB (piso digital)')


#####################################################################
# SECCION 2: PUNTO A (B = 4 bits, kn = 1.0)
#####################################################################

res_a = procesar_adc(B=4, kn=1.0)

print("="*65)
print(">>> RESULTADOS PUNTO A (B = 4 bits, kn = 1.0) <<<")
print(f"Paso de cuantizacion q:             {res_a['q']:.4f} V")
print(f"Potencia teorica Pq:                {res_a['Pq']:.6f} W")
print(f"Potencia teorica Pn:                {res_a['Pn']:.6f} W")
print(f"-> Piso Ruido Digital Teorico:      {res_a['piso_digital']:.2f} dB")
print(f"-> Piso Ruido Analogico Teorico:    {res_a['piso_analog']:.2f} dB")
print(f"   (Nota: Al ser kn=1, ambos pisos coinciden exactamente en {res_a['piso_digital']:.2f} dB)")
print(f"SNR antes del ADC:                  {res_a['SNR_antes']:.2f} dB")
print(f"SNR despues del ADC:                {res_a['SNR_despues']:.2f} dB")
print(f"Caida de SNR por digitalizacion:    {(res_a['SNR_antes'] - res_a['SNR_despues']):.2f} dB")
print("="*65 + "\n")

# FIGURA 1: Senales temporales
plt.figure(1, figsize=(10, 4.5))
plt.plot(tt, res_a['s_cuantizada'], label=r'$s_Q = Q_{B, V_R}(s_R)$ (ADC out)', color='C0', lw=1.2)
plt.plot(tt, res_a['sR'], label=r'$s_R = s + n$ (ADC in)', color='green', linestyle=':', alpha=0.7)
plt.plot(tt, s, label=r'$s$ (analog)', color='orange', linestyle=':', alpha=0.7)
plt.title(f'Señal muestreada por un ADC de 4 bits - $\\pm V_R = {VF:.1f}$ V - q = {res_a["q"]:.3f} V')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()

# FIGURA 2: Densidad de Potencia
plt.figure(2, figsize=(10, 4.5))
plt.plot(ff, res_a['S_out_db'], label=r'$s_Q = Q_{B, V_R}(s_R)$ (ADC out)', color='C0', lw=0.9)
plt.plot(ff, S_orig_db, label=r'$s$ (analog)', color='orange', linestyle=':', alpha=0.7)
plt.plot(ff, res_a['S_in_db'], label=r'$s_R = s + n$ (ADC in)', color='green', linestyle=':', alpha=0.7)
graficar_pisos(plt.gca(), res_a['piso_analog'], res_a['piso_digital'])
plt.title(f'Señal muestreada por un ADC de 4 bits - $\\pm V_R = {VF:.1f}$ V - q = {res_a["q"]:.3f} V')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlim(0, fs/2)
plt.ylim(-85, 5)
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()

# FIGURA 3: Histograma
plt.figure(3, figsize=(10, 4.5))
bins_hist = np.linspace(-res_a['q']/2, res_a['q']/2, 11)
plt.hist(res_a['ruido_cuantizacion'], bins=bins_hist, color='C0', edgecolor='white', lw=0.5)
altura_teorica = N / 10
plt.plot([-res_a['q']/2, -res_a['q']/2, res_a['q']/2, res_a['q']/2], [0, altura_teorica, altura_teorica, 0],
         color='red', linestyle='--', lw=1.5, label='Distribucion teorica')
plt.title(f'Ruido de cuantización para 4 bits - $\\pm V_R = {VF:.1f}$ V - q = {res_a["q"]:.3f} V')
plt.xlim(-res_a['q']/2 * 1.1, res_a['q']/2 * 1.1)
plt.ylim(0, altura_teorica * 1.25)
plt.grid(False)
plt.tight_layout()


#####################################################################
# SECCION 3: PUNTO B - COMPARACION PARA B = 4 BITS (kn = 0.1, 1, 10)
#####################################################################

kn_valores = [0.1, 1.0, 10.0]

print("="*65)
print(">>> PUNTO B: CONFIGURACIONES PARA B = 4 BITS <<<")
fig4, axs4 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

for idx, kn in enumerate(kn_valores):
    res = procesar_adc(B=4, kn=kn)
    ax = axs4[idx]
    
    print(f"Configuracion: B = 4 bits | kn = {kn:<4}")
    print(f"  Piso analogico: {res['piso_analog']:>7.2f} dB | Piso digital: {res['piso_digital']:>7.2f} dB")
    print(f"  SNR antes:      {res['SNR_antes']:>7.2f} dB | SNR despues:  {res['SNR_despues']:>7.2f} dB | Caida: {(res['SNR_antes']-res['SNR_despues']):.2f} dB")
    
    ax.plot(ff, res['S_out_db'], label=r'$s_Q$ (ADC out)', color='C0', lw=0.8)
    ax.plot(ff, S_orig_db, label=r'$s$ (analog)', color='orange', linestyle=':', alpha=0.7)
    ax.plot(ff, res['S_in_db'], label=r'$s_R$ (ADC in)', color='green', linestyle=':', alpha=0.6)
    graficar_pisos(ax, res['piso_analog'], res['piso_digital'])
    
    ax.set_title(f'B = 4 bits | kn = {kn} | (q = {res["q"]:.3f} V)')
    ax.set_ylabel('Potencia [dB]')
    ax.set_ylim(-90, 5)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower left', fontsize=8)

axs4[-1].set_xlabel('Frecuencia [Hz]')
axs4[-1].set_xlim(0, fs/2)
fig4.suptitle('Punto B: Comparación para B = 4 bits variando kn', fontsize=12, y=0.99)
plt.tight_layout()
print("="*65 + "\n")


#####################################################################
# SECCION 4: PUNTO B - COMPARACION PARA B = 8 BITS (kn = 0.1, 1, 10)
#####################################################################

print("="*65)
print(">>> PUNTO B: CONFIGURACIONES PARA B = 8 BITS <<<")
fig5, axs5 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

for idx, kn in enumerate(kn_valores):
    res = procesar_adc(B=8, kn=kn)
    ax = axs5[idx]
    
    print(f"Configuracion: B = 8 bits | kn = {kn:<4}")
    print(f"  Piso analogico: {res['piso_analog']:>7.2f} dB | Piso digital: {res['piso_digital']:>7.2f} dB")
    print(f"  SNR antes:      {res['SNR_antes']:>7.2f} dB | SNR despues:  {res['SNR_despues']:>7.2f} dB | Caida: {(res['SNR_antes']-res['SNR_despues']):.2f} dB")
    
    ax.plot(ff, res['S_out_db'], label=r'$s_Q$ (ADC out)', color='C0', lw=0.8)
    ax.plot(ff, S_orig_db, label=r'$s$ (analog)', color='orange', linestyle=':', alpha=0.7)
    ax.plot(ff, res['S_in_db'], label=r'$s_R$ (ADC in)', color='green', linestyle=':', alpha=0.6)
    graficar_pisos(ax, res['piso_analog'], res['piso_digital'])
    
    ax.set_title(f'B = 8 bits | kn = {kn} | (q = {res["q"]:.4e} V)')
    ax.set_ylabel('Potencia [dB]')
    ax.set_ylim(-115, 5)        # Rango adaptado a 8 bits
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower left', fontsize=8)

axs5[-1].set_xlabel('Frecuencia [Hz]')
axs5[-1].set_xlim(0, fs/2)
fig5.suptitle('Punto B: Comparación para B = 8 bits variando kn', fontsize=12, y=0.99)
plt.tight_layout()
print("="*65 + "\n")


#####################################################################
# SECCION 5: PUNTO B - COMPARACION PARA B = 16 BITS (kn = 0.1, 1, 10)
#####################################################################

print("="*65)
print(">>> PUNTO B: CONFIGURACIONES PARA B = 16 BITS <<<")
fig6, axs6 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

for idx, kn in enumerate(kn_valores):
    res = procesar_adc(B=16, kn=kn)
    ax = axs6[idx]
    
    print(f"Configuracion: B = 16 bits | kn = {kn:<4}")
    print(f"  Piso analogico: {res['piso_analog']:>7.2f} dB | Piso digital: {res['piso_digital']:>7.2f} dB")
    print(f"  SNR antes:      {res['SNR_antes']:>7.2f} dB | SNR despues:  {res['SNR_despues']:>7.2f} dB | Caida: {(res['SNR_antes']-res['SNR_despues']):.2f} dB")
    
    ax.plot(ff, res['S_out_db'], label=r'$s_Q$ (ADC out)', color='C0', lw=0.8)
    ax.plot(ff, S_orig_db, label=r'$s$ (analog)', color='orange', linestyle=':', alpha=0.7)
    ax.plot(ff, res['S_in_db'], label=r'$s_R$ (ADC in)', color='green', linestyle=':', alpha=0.6)
    graficar_pisos(ax, res['piso_analog'], res['piso_digital'])
    
    ax.set_title(f'B = 16 bits | kn = {kn} | (q = {res["q"]:.4e} V)')
    ax.set_ylabel('Potencia [dB]')
    ax.set_ylim(-165, 5)        # Rango adaptado a 16 bits
    ax.grid(True, alpha=0.4)
    ax.legend(loc='lower left', fontsize=8)

axs6[-1].set_xlabel('Frecuencia [Hz]')
axs6[-1].set_xlim(0, fs/2)
fig6.suptitle('Punto B: Comparación para B = 16 bits variando kn', fontsize=12, y=0.99)
plt.tight_layout()
print("="*65 + "\n")

# Mostrar todas las figuras
plt.show()

#####################################################################
# SECCION 6: FIGURA COMPARATIVA FINAL (4 vs 8 vs 16 BITS para kn = 1.0)
#####################################################################

kn_ref = 1.0
bits_comp = [4, 8, 16]
colores = {4: 'tab:blue', 8: 'tab:orange', 16: 'tab:green'}

fig7, (ax_esp, ax_temp) = plt.subplots(2, 1, figsize=(11, 8.5))

# Zoom temporal para ver los escalones (primeros 80 ms = 80 muestras)
zoom_samples = slice(0, 80)

# Señal analogica continua como referencia en el grafico temporal
ax_temp.plot(tt[zoom_samples], s[zoom_samples], label='s (analógica original)',
             color='black', linestyle='--', lw=1.5, alpha=0.8)

for B_val in bits_comp:
    res = procesar_adc(B=B_val, kn=kn_ref)
    c = colores[B_val]
    
    # 1. PANEL ESPECTRAL: Superposicion de los 3 espectros
    ax_esp.plot(ff, res['S_out_db'], color=c, alpha=0.75, lw=0.9,
                label=f'B = {B_val:2d} bits  (Piso: {res["piso_digital"]:.1f} dB)')
    ax_esp.axhline(res['piso_digital'], color=c, linestyle=':', lw=1.2)
    
    # 2. PANEL TEMPORAL: Superposicion de los escalones de cuantizacion
    ax_temp.step(tt[zoom_samples], res['s_cuantizada'][zoom_samples],
                 where='mid', color=c, lw=1.2,
                 label=f'B = {B_val:2d} bits  (q = {res["q"]*1000:6.2f} mV)')

# Configuracion del panel de espectros
ax_esp.set_title(f'Comparación en Frecuencia: Caída del Piso de Ruido Digital (kn = {kn_ref})', fontsize=11)
ax_esp.set_xlabel('Frecuencia [Hz]')
ax_esp.set_ylabel('Densidad de Potencia [dB]')
ax_esp.set_xlim(0, fs/2)
ax_esp.set_ylim(-160, 8)
ax_esp.grid(True, alpha=0.4)
ax_esp.legend(loc='lower left', fontsize=9)

# Configuracion del panel temporal
ax_temp.set_title('Comparación en Tiempo: Resolución de los Escalones de Cuantización (Zoom 0 a 0.08 s)', fontsize=11)
ax_temp.set_xlabel('Tiempo [segundos]')
ax_temp.set_ylabel('Amplitud [V]')
ax_temp.grid(True, alpha=0.4)
ax_temp.legend(loc='upper right', fontsize=9)

fig7.suptitle('Figura 7: Impacto Directo de la Cantidad de Bits (B = 4, 8 y 16) con kn = 1.0', fontsize=13, y=0.99)
plt.tight_layout()
