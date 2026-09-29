#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis y Procesamiento de Señales (APS)
TS5 — Estimación de la Densidad Espectral de Potencia (PSD)
Autor: Pedro Joaquín Cannavo
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

# ============================================================
#  1. LECTURA DE SEÑALES
# ============================================================

# ── ECG sin ruido ──────────────────────────────────────────
fs_ecg = 1000  # Hz
ecg = np.load('ecg_sin_ruido.npy').flatten()
t_ecg = np.arange(len(ecg)) / fs_ecg

# ── PPG sin ruido ──────────────────────────────────────────
fs_ppg = 400  # Hz
ppg = np.load('ppg_sin_ruido.npy').flatten()
t_ppg = np.arange(len(ppg)) / fs_ppg

# ── Audio (La Cucaracha) ───────────────────────────────────
fs_audio, audio_raw = sio.wavfile.read('la cucaracha.wav')

# Convertir a mono si es multicanal y normalizar a float en [-1, 1]
if audio_raw.ndim > 1:
    audio = audio_raw[:, 0].astype(np.float64)
else:
    audio = audio_raw.astype(np.float64)

audio = audio / np.max(np.abs(audio))
t_audio = np.arange(len(audio)) / fs_audio

print(f'ECG   -> {len(ecg)} muestras | Duración: {len(ecg)/fs_ecg:.1f} s | fs = {fs_ecg} Hz')
print(f'PPG   -> {len(ppg)} muestras | Duración: {len(ppg)/fs_ppg:.1f} s | fs = {fs_ppg} Hz')
print(f'Audio -> {len(audio)} muestras | Duración: {len(audio)/fs_audio:.1f} s | fs = {fs_audio} Hz')


# ============================================================
#  2. VISUALIZACIÓN TEMPORAL
# ============================================================

fig, axs = plt.subplots(3, 1, figsize=(12, 8))
fig.suptitle('Señales en el dominio del tiempo', fontsize=13)

seg_ecg = min(10 * fs_ecg, len(ecg))
axs[0].plot(t_ecg[:seg_ecg], ecg[:seg_ecg], color='steelblue', linewidth=0.8)
axs[0].set_title(f'ECG — Prueba de esfuerzo (fs = {fs_ecg} Hz)')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud [mV]')
axs[0].grid(True, alpha=0.4)

axs[1].plot(t_ppg, ppg, color='darkorange', linewidth=0.8)
axs[1].set_title(f'PPG — Registro en reposo (fs = {fs_ppg} Hz)')
axs[1].set_xlabel('Tiempo [s]')
axs[1].set_ylabel('Amplitud [u.a.]')
axs[1].grid(True, alpha=0.4)

seg_audio = min(5 * fs_audio, len(audio))
axs[2].plot(t_audio[:seg_audio], audio[:seg_audio], color='seagreen', linewidth=0.5)
axs[2].set_title(f'Audio — La Cucaracha (fs = {fs_audio} Hz)')
axs[2].set_xlabel('Tiempo [s]')
axs[2].set_ylabel('Amplitud norm.')
axs[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()


# ============================================================
#  3. EXPLORACIÓN DE PARÁMETROS DE WELCH
# ============================================================

def welch_comparacion(señal, fs, configs, titulo, ax):
    for cfg in configs:
        f, Pxx = sig.welch(
            señal, fs=fs, window='hann',
            nperseg=cfg['nperseg'],
            noverlap=cfg['noverlap'],
            scaling='density'
        )
        step  = cfg['nperseg'] - cfg['noverlap']
        n_avg = 1 + (len(señal) - cfg['nperseg']) // step
        df    = fs / cfg['nperseg']
        Pxx_dB = 10 * np.log10(Pxx + 1e-12)
        label = f"{cfg['label']}  (K≈{n_avg}, Δf≈{df:.2f} Hz)"
        ax.plot(f, Pxx_dB, label=label, color=cfg['color'], linewidth=0.9)

    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('PSD [dB/Hz]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

configs_ecg = [
    {'nperseg': len(ecg)//2, 'noverlap': len(ecg)//4, 'label': '① Pocos promedios (L=N/2)', 'color': 'crimson'},
    {'nperseg': 2048,        'noverlap': 1024,         'label': '② Óptima (L=2048)',          'color': 'royalblue'},
    {'nperseg': 128,         'noverlap': 64,           'label': '③ Excesiva (L=128)',          'color': 'forestgreen'},
]
configs_ppg = [
    {'nperseg': len(ppg)//2, 'noverlap': len(ppg)//4, 'label': '① Pocos promedios (L=N/2)', 'color': 'crimson'},
    {'nperseg': 1024,        'noverlap': 512,          'label': '② Óptima (L=1024)',          'color': 'royalblue'},
    {'nperseg': 64,          'noverlap': 32,           'label': '③ Excesiva (L=64)',           'color': 'forestgreen'},
]
configs_audio = [
    {'nperseg': len(audio)//2, 'noverlap': len(audio)//4, 'label': '① Pocos promedios (L=N/2)', 'color': 'crimson'},
    {'nperseg': 8192,          'noverlap': 4096,           'label': '② Óptima (L=8192)',          'color': 'royalblue'},
    {'nperseg': 256,           'noverlap': 128,            'label': '③ Excesiva (L=256)',          'color': 'forestgreen'},
]

fig_comp, axs_comp = plt.subplots(3, 1, figsize=(13, 12))
fig_comp.suptitle('Exploración del compromiso resolución vs. varianza en Welch', fontsize=13)

welch_comparacion(ecg,   fs_ecg,   configs_ecg,   f'ECG (fs={fs_ecg} Hz)',    axs_comp[0])
welch_comparacion(ppg,   fs_ppg,   configs_ppg,   f'PPG (fs={fs_ppg} Hz)',    axs_comp[1])
welch_comparacion(audio, fs_audio, configs_audio, f'Audio (fs={fs_audio} Hz)', axs_comp[2])

axs_comp[0].set_xlim(0, 100)
axs_comp[1].set_xlim(0, 20)
axs_comp[2].set_xlim(0, fs_audio // 2)

plt.tight_layout()
plt.show()


# ============================================================
#  4. ESTIMACIÓN FINAL DE LA PSD (CONFIGURACIÓN ÓPTIMA)
# ============================================================

f_ecg, P_ecg = sig.welch(
    ecg, fs=fs_ecg, window='hann',
    nperseg=2048, noverlap=1024, scaling='density'
)
f_ppg, P_ppg = sig.welch(
    ppg, fs=fs_ppg, window='hann',
    nperseg=1024, noverlap=512, scaling='density'
)
f_audio, P_audio = sig.welch(
    audio, fs=fs_audio, window='hann',
    nperseg=8192, noverlap=4096, scaling='density'
)

P_ecg_dB   = 10 * np.log10(P_ecg   + 1e-12)
P_ppg_dB   = 10 * np.log10(P_ppg   + 1e-12)
P_audio_dB = 10 * np.log10(P_audio + 1e-12)


# ============================================================
#  5. ESTIMACIÓN DEL ANCHO DE BANDA (BW 99.5%)
# ============================================================

def estimar_ancho_de_banda(f, P, porcentaje=0.995):
    """
    Calcula la frecuencia de corte hasta la cual se concentra el porcentaje
    especificado (por defecto 99.5%) de la energía total de la PSD.
    """
    potencia_acum = np.cumsum(P)
    potencia_total = potencia_acum[-1]
    idx_bw = np.searchsorted(potencia_acum / potencia_total, porcentaje)
    bw = f[min(idx_bw, len(f) - 1)]
    return bw, potencia_total

bw_ecg,   _ = estimar_ancho_de_banda(f_ecg,   P_ecg,   porcentaje=0.995)
bw_ppg,   _ = estimar_ancho_de_banda(f_ppg,   P_ppg,   porcentaje=0.995)
bw_audio, _ = estimar_ancho_de_banda(f_audio, P_audio, porcentaje=0.995)

print('\nANCHO DE BANDA ESTIMADO (Criterio 99.5% de energía):')
print('─' * 70)
print(f'[ECG]        | BW = {bw_ecg:8.2f} Hz  (desde 0.00 Hz hasta {bw_ecg:.2f} Hz)')
print(f'[PPG]        | BW = {bw_ppg:8.2f} Hz  (desde 0.00 Hz hasta {bw_ppg:.2f} Hz)')
print(f'[Audio]      | BW = {bw_audio:8.2f} Hz  (desde 0.00 Hz hasta {bw_audio:.2f} Hz)')
print('─' * 70)


# ────────────────────────────────────────────────────────────
# Gráficos finales con línea de BW
# ────────────────────────────────────────────────────────────

fig_psd, axs_psd = plt.subplots(3, 1, figsize=(13, 12))
fig_psd.suptitle('PSD Final — Método de Welch con Ventana Hann', fontsize=13)

# ECG
axs_psd[0].plot(f_ecg, P_ecg_dB, color='steelblue', linewidth=1)
axs_psd[0].axvline(bw_ecg, color='red', linestyle='--', linewidth=1.2,
                   label=f'BW (99.5%) = {bw_ecg:.1f} Hz')
axs_psd[0].set_title(f'ECG  |  fs = {fs_ecg} Hz  |  L = 2048  |  Δf ≈ {fs_ecg/2048:.2f} Hz')
axs_psd[0].set_xlabel('Frecuencia [Hz]')
axs_psd[0].set_ylabel('PSD [dB/Hz]')
axs_psd[0].set_xlim(0, 100)
axs_psd[0].legend(fontsize=9)
axs_psd[0].grid(True, alpha=0.4)

# PPG
axs_psd[1].plot(f_ppg, P_ppg_dB, color='darkorange', linewidth=1)
axs_psd[1].axvline(bw_ppg, color='red', linestyle='--', linewidth=1.2,
                   label=f'BW (99.5%) = {bw_ppg:.1f} Hz')
axs_psd[1].set_title(f'PPG  |  fs = {fs_ppg} Hz  |  L = 1024  |  Δf ≈ {fs_ppg/1024:.2f} Hz')
axs_psd[1].set_xlabel('Frecuencia [Hz]')
axs_psd[1].set_ylabel('PSD [dB/Hz]')
axs_psd[1].set_xlim(0, 20)
axs_psd[1].legend(fontsize=9)
axs_psd[1].grid(True, alpha=0.4)

# Audio
axs_psd[2].plot(f_audio, P_audio_dB, color='seagreen', linewidth=0.8)
axs_psd[2].axvline(bw_audio, color='red', linestyle='--', linewidth=1.2,
                   label=f'BW (99.5%) = {bw_audio:.1f} Hz')
axs_psd[2].set_title(f'Audio (La Cucaracha)  |  fs = {fs_audio} Hz  |  L = 8192  |  Δf ≈ {fs_audio/8192:.1f} Hz')
axs_psd[2].set_xlabel('Frecuencia [Hz]')
axs_psd[2].set_ylabel('PSD [dB/Hz]')
axs_psd[2].set_xlim(0, fs_audio // 2)
axs_psd[2].legend(fontsize=9)
axs_psd[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()


# ============================================================
#  6. TABLA RESUMEN
# ============================================================

print('\n' + '=' * 72)
print(f'{"TABLA RESUMEN — ANCHOS DE BANDA ESTIMADOS":<72}')
print('=' * 72)
print(f'{"Señal":<22} {"fs [Hz]":>9} {"L (nperseg)":>12} {"Δf [Hz]":>9} {"BW_99.5% [Hz]":>14}')
print('-' * 72)
print(f'{"ECG":<22} {fs_ecg:>9} {2048:>12} {fs_ecg/2048:>9.3f} {bw_ecg:>14.2f}')
print(f'{"PPG":<22} {fs_ppg:>9} {1024:>12} {fs_ppg/1024:>9.3f} {bw_ppg:>14.2f}')
print(f'{"Audio (La Cucaracha)":<22} {fs_audio:>9} {8192:>12} {fs_audio/8192:>9.2f} {bw_audio:>14.2f}')
print('=' * 72)
print('Criterio de BW: frecuencia acumulada al 99.5% de la energía total.')
print('Método: Welch con ventana Hann y 50% de solapamiento.\n')
