#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:37:14 2021

@author: nacho
"""

from sistema_de_agentes import Replicadores_parentesco
import numpy as np
from matplotlib import pyplot as plt
from medidores import Media, Desvio_Estandar, Histo_Log, Histo_Lineal
import os

def simulacion(N_total, mean_hijes, pasos = 10**6,
               q=0.1, lamda=1.0, u_adim=1e-2, a=1.0, dt=1e-3,
               k_list=range(1,8)):
    """Simulación de un sistema de  N_total replicadores emparentados.
    
    Devuelve un diccionario con el tamaño del sistema, un histograma de los
    valores totales de recursos visitados, promedio temporal acumulado y
    fluctuaciones temporales.
    """
    
    sistema = Replicadores_parentesco(
        N_total,
        mean_hijes,
        x0 = 1.0 / N_total / mean_hijes,
        q = q,
        lamda = lamda,
        u = u_adim,
        a = a, 
        dt = dt
        )

    sistema.transitorio(pasos=10000)
    
    histo_media = Histo_Lineal(
        xmin = u_adim / sistema.n,
        xmax = 1.0 / sistema.n,
        nbins = 100
        )
    histo_recursos = Histo_Log(
        xmin = u_adim / sistema.n,
        xmax = 1.0 / sistema.n,
        nbins = 100
        )
    
    # diccionario de histogramas por numero de vecinos
    histo_vecinos = {i: Histo_Lineal(
        xmin = u_adim / sistema.n,
        xmax = 1.0 / sistema.n,
        nbins = 100
        ) for i in k_list}
 
    for i in range(pasos):
        sistema.step()
        if not i % 1000:
            mu = sistema.mean()
            histo_media.nuevo_dato(mu)
            histo_recursos.nuevo_dato(sistema.x)
            dict_vecinos = sistema.red.segregate_per_neighbours(k_list)
            for n, histo in list(
                    zip(dict_vecinos.keys(), dict_vecinos.values())):
                histo_vecinos[n].nuevo_dato(sistema.x[dict_vecinos[n]])
            print(f"ETA: ----- {100.0 * i / pasos:.2f}%")
    
    return {"N": sistema.n,
            "histo_media": histo_media,
            "histo_recursos": histo_recursos,
            "histos_vecinos": histo_vecinos
            }

#%% Simulaciones
#----------Parámetros de la red
N_total = 100 #numero total de agentes

hijos_mean = 2
#----------
dic = simulacion(N_total, hijos_mean, a=1)
histo_media = dic['histo_media']
histo_recursos = dic['histo_recursos']
histo_recursos.plot_densidad(scale='log')
histo_media = dic['histo_media']
histo_media.plot_densidad(scale='log')
plt.legend(["Recursos", "Media"], loc='best')
plt.figure()
for histo in dic["histos_vecinos"].values():
    histo.plot_densidad(scale='log')
plt.legend([f"$k={i:d}$" for i in dic["histos_vecinos"]], loc='best')