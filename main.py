#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:37:14 2021

@author: nacho
"""

from sistema_de_agentes import Replicadores_parentesco
import numpy as np
from matplotlib import pyplot as plt
from medidores import Media, Desvio_Estandar, Histo_Log
import os

def simulacion(M, mean_hijes, q=0.1, lamda=1.0, u_adim=1e-2, a=1.0, dt=1e-3):
    """Simulación de un sistema de replicadores emparentados.
    
    Devuelve un diccionario con el tamaño del sistema, un histograma de los
    valores totales de recursos visitados, promedio temporal acumulado y
    fluctuaciones temporales.
    """
    
    sistema = Replicadores_parentesco(
        M,
        mean_hijes,
        x0 = 1.0 / M / mean_hijes,
        q = q,
        lamda = lamda,
        u = u_adim,
        a = a, 
        dt = dt
        )
    sistema.transitorio(pasos=10000)
    
    bar_x_t = Media()
    sigma_bar_x_t = Desvio_Estandar()
    histo = Histo_Log(
        xmin = u_adim / sistema.n,
        xmax = 1.0 / sistema.n,
        nbins = 100
        )
    pasos = 10 ** 6
    
    for i in range(pasos):
        sistema.step()
        mu = sistema.mean()
        bar_x_t.nuevo_dato(mu)
        sigma_bar_x_t.nuevo_dato(mu)
        if not i % 1000:
            histo.nuevo_dato(sistema.mean())
            print(f"ETA: ----- {100.0 * i / pasos:.2f}%")
    
    return {"N": sistema.n,
            "histo": histo,
            "bar_x_t": bar_x_t,
            "sigma_bar_x_t": sigma_bar_x_t
            }

#%% Simulaciones
#----------Parámetros de la red
M = 50
hijos_mean = 2
#----------
archivos = os.listdir("../Agentes_replicadores/data/")
experimentos = [os.path.join("../Agentes_replicadores/data/", archivo)
                for archivo in archivos if "density_N" in archivo]
experimentos.sort(reverse=True)

for exp in experimentos:
    x, f = np.loadtxt(exp, skiprows=1).T
    N = int(exp[exp.find("N")+1:exp.find("_q")])
    if N < 10**5:
        plt.loglog(
            x[f>0] * N,
            f[f>0] / N,
            #marker='.',
            alpha=0.9,
            linewidth = 1,
            label=f'N=$10^{int(np.log10(N)):d}$'
            )

plt.legend(loc='best')
plt.xticks([1e-2, 1e-1, 1], [r"$N \; u$", r"$0.1$", r"$1$"])
plt.xlabel(r"$N \bar x $", fontsize=13)
plt.ylabel(r"$f(\bar x) / N$", fontsize=13, rotation = 0, y=0.6)