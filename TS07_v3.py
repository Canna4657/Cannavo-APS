import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal as sig

# ============================================================
# 1. CARGA DE DATOS Y CONFIGURACIÓN INICIAL
# ============================================================
fs = 1000  # Frecuencia de muestreo [Hz]
mat = sio.loadmat('./ecg.mat')

ecg_one_lead = mat['ecg_lead'].flatten().astype(float)
qrs_det = mat['qrs_detections'].flatten()
hb_1 = mat['heartbeat_pattern1'].flatten()
hb_2 = mat['heartbeat_pattern2'].flatten()

nyq_frec = fs / 2
ripple = 1.0  # dB (atenuación máxima en banda de paso)
atenuacion = 40.0  # dB (atenuación mínima en banda de rechazo)

# Especificación de la plantilla (Filtro Pasa-Banda)
ws1, wp1 = 0.1, 0.8  # Frecuencias de corte inferior [Hz]
wp2, ws2 = 31.5, 45.0  # Frecuencias de corte superior [Hz]

wp = [wp1, wp2]
ws = [ws1, ws2]

# ============================================================
# 2. DISEÑO Y EVALUACIÓN DE FILTROS IIR
# ============================================================
sos_butt = sig.iirdesign(
    wp, ws, gpass=ripple, gstop=atenuacion, ftype='butter', output='sos', fs=fs
)
sos_cheb = sig.iirdesign(
    wp, ws, gpass=ripple, gstop=atenuacion, ftype='cheby1', output='sos', fs=fs
)
sos_cauer = sig.iirdesign(
    wp, ws, gpass=ripple, gstop=atenuacion, ftype='ellip', output='sos', fs=fs
)

filtros_iir = {
    'Butterworth': sos_butt,
    'Chebyshev I': sos_cheb,
    'Cauer': sos_cauer,
}
colores = {
    'Butterworth': 'tab:blue',
    'Chebyshev I': 'tab:orange',
    'Cauer': 'tab:green',
}

print('--- ORDEN DE LOS FILTROS IIR ---')
for nombre, sos in filtros_iir.items():
  print(f'{nombre}: {sos.shape[0]} secciones -> orden {2*sos.shape[0]}')

# Vectores de frecuencia para análisis
f = np.logspace(-2, np.log10(nyq_frec), 5000)
respuestas_iir = {}
for nombre, sos in filtros_iir.items():
  w, H = sig.sosfreqz(sos, worN=f, fs=fs)
  respuestas_iir[nombre] = H


# Función auxiliar para dibujar las zonas prohibidas de la plantilla
def dibujar_plantilla(ax, f_min, f_max, y_min=-80, y_max=5):
  ax.fill_between([f_min, ws1], -atenuacion, y_max, color='red', alpha=0.15)
  ax.fill_between([wp1, wp2], y_min, -ripple, color='red', alpha=0.15)
  ax.fill_between(
      [ws2, f_max],
      -atenuacion,
      y_max,
      color='red',
      alpha=0.15,
      label='Zona prohibida (Plantilla)',
  )


# Gráfico de Módulo IIR (Eje logarítmico)
fig, ax = plt.subplots(figsize=(10, 4.5))
for nombre, H in respuestas_iir.items():
  ax.semilogx(
      f, 20 * np.log10(np.abs(H) + 1e-12), label=nombre, color=colores[nombre]
  )
dibujar_plantilla(ax, f[0], nyq_frec)
ax.set_xlim(f[0], nyq_frec)
ax.set_ylim(-80, 5)
ax.set_title('Módulo de los IIR vs Plantilla (Banda Pasante)')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Módulo [dB]')
ax.grid(True, which='both', linestyle=':')
ax.legend(loc='lower left')
plt.tight_layout()
plt.show()

# ============================================================
# 3. DISEÑO Y EVALUACIÓN DE FILTROS FIR
# ============================================================
cant_coef = 1501  # Orden óptimo y razonable para los FIR (Tipo I, impar)
demora = (cant_coef - 1) // 2

# Diseñamos un FIR por método de Ventana (Kaiser) y otro por Mínimos Cuadrados (firls)
c1, c2 = (ws1 + wp1) / 2, (wp2 + ws2) / 2
h_win = sig.firwin2(
    cant_coef,
    [0, c1, c1, c2, c2, nyq_frec],
    [0, 0, 1, 1, 0, 0],
    window=('kaiser', 4.0),
    fs=fs,
)
h_ls = sig.firls(
    cant_coef,
    [0, ws1, ws1, wp1, wp1, wp2, wp2, ws2, ws2, nyq_frec],
    [0, 0,   0,   1,   1,   1,   1,   0,   0,   0],
    weight=[1, 0.01, 1, 0.01, 1],
    fs=fs,
)

filtros_fir = {'FIR ventana': h_win, 'FIR firls': h_ls}
colores_fir = {'FIR ventana': 'tab:red', 'FIR firls': 'tab:purple'}

print('\n--- INFORMACIÓN DE LOS FILTROS FIR ---')
for nombre, h in filtros_fir.items():
  print(
      f'{nombre}: {len(h)} coeficientes, retardo = {demora} muestras ='
      f' {demora/fs:.3f} s'
  )

respuestas_fir = {}
for nombre, h in filtros_fir.items():
  w, H = sig.freqz(h, worN=f, fs=fs)
  respuestas_fir[nombre] = H

# Gráfico de Módulo FIR
fig, ax = plt.subplots(figsize=(10, 4.5))
for nombre, H in respuestas_fir.items():
  ax.semilogx(
      f,
      20 * np.log10(np.abs(H) + 1e-12),
      label=nombre,
      color=colores_fir[nombre],
  )
dibujar_plantilla(ax, f[0], nyq_frec)
ax.set_xlim(f[0], nyq_frec)
ax.set_ylim(-80, 5)
ax.set_title('Módulo de los FIR vs Plantilla')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Módulo [dB]')
ax.grid(True, which='both', linestyle=':')
ax.legend(loc='lower left')
plt.tight_layout()
plt.show()

# ============================================================
# 4. APLICACIÓN DE LOS FILTROS AL ECG
# ============================================================
# IIR con fase cero (sosfiltfilt)
ecg_butt_ff = sig.sosfiltfilt(sos_butt, ecg_one_lead)
ecg_cheb_ff = sig.sosfiltfilt(sos_cheb, ecg_one_lead)
ecg_cauer_ff = sig.sosfiltfilt(sos_cauer, ecg_one_lead)


# FIR compensando la demora lineal (fase lineal perfecta)
def filtrar_fir(h, x):
  d = (len(h) - 1) // 2
  y = sig.oaconvolve(x, h, mode='full')[: len(x) + d]
  return y[d:]  # Desplaza y alinea temporalmente


ecg_win = filtrar_fir(h_win, ecg_one_lead)
ecg_ls = filtrar_fir(h_ls, ecg_one_lead)

# Para graficar los resultados fácilmente
senales = {
    'Butterworth': (ecg_butt_ff, 'tab:blue'),
    'Chebyshev I': (ecg_cheb_ff, 'tab:orange'),
    'Cauer': (ecg_cauer_ff, 'tab:green'),
    'FIR ventana': (ecg_win, 'tab:red'),
    'FIR firls': (ecg_ls, 'tab:purple'),
}

# ============================================================
# 5. GRÁFICOS DE ZOOMS (Zonas limpias y con ruido)
# ============================================================


def graficar_regiones(regiones, titulo_grupo):
  cant = ecg_one_lead.shape[0]
  for ini, fin in regiones:
    ini, fin = int(max(0, ini)), int(min(cant, fin))
    if fin <= ini:
      continue
    zoom = np.arange(ini, fin)

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # ECG original de fondo en ambos paneles
    for ax in axs:
      ax.plot(
          zoom,
          ecg_one_lead[zoom],
          label='ECG original',
          color='gray',
          linewidth=1.8,
          alpha=0.7,
      )

    # Ploteamos IIR arriba y FIR abajo
    for nombre, (x, col) in senales.items():
      ax = axs[0] if nombre in filtros_iir else axs[1]
      ax.plot(
          zoom, x[zoom], label=nombre, color=col, linewidth=1.2, alpha=0.9
      )

    axs[0].set_title(
        f'{titulo_grupo} - Filtros IIR (Muestras {ini} a {fin})',
        fontweight='bold',
    )
    axs[1].set_title(
        f'{titulo_grupo} - Filtros FIR (Muestras {ini} a {fin})',
        fontweight='bold',
    )
    axs[1].set_xlabel('Muestras (#)')

    for ax in axs:
      ax.legend(loc='upper right', fontsize=9)
      ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()


# A. Regiones estables / sin ruido (ej: tramos intermedios limpios)
regs_limpias = ([4000, 5500], [10000, 11000])
graficar_regiones(regs_limpias, 'Zonas Estables (Sin ruido evidente)')

# B. Regiones con ruido e interferencias (convertidas de minutos a muestras)
regs_ruido = [
    np.array([5.0, 5.2]) * 60 * fs,
    np.array([12.0, 12.4]) * 60 * fs,
]
graficar_regiones(regs_ruido, 'Zonas Críticas / Con Ruido')