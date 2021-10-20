#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:37:14 2021

@author: nacho
"""

from medidores import Media, Desvio_Estandar, Histo_Lineal, Histo_Log
from sistema_de_agentes import Replicadores, Desacoplados
import numpy as np
from matplotlib import pyplot as plt
from funciones import f_cont
import os

n = 1000
def simulacion(n):
    x0 = np.ones(n)
    q = 0.1 * np.ones(n)
    lamda = np.ones(n)
    u = 0.1 * np.ones(n)# / (10*n)
    
    replicadores = Replicadores(n, x0/n, q, lamda, u/n, dt=1e-3)
    replicadores.transitorio(pasos=10000)
    
    bar_x_t = Media()
    sigma_bar_x_t = Desvio_Estandar()
    histo = Histo_Log(xmin = (u/n).min(), xmax = 1.0/n, nbins = 100)
    pasos = 10 ** 6
    
    for i in range(pasos):
        replicadores.step()
        mu = replicadores.mean()
        bar_x_t.nuevo_dato(mu)
        sigma_bar_x_t.nuevo_dato(mu)
        if not i % 5:
            histo.nuevo_dato(replicadores.mean()) #ojo, si hay muchos agentes, hay cond. carrera
    
    return {"N": n,
            "histo": histo,
            "bar_x_t": bar_x_t,
            "sigma_bar_x_t": sigma_bar_x_t
            }

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