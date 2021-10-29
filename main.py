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
    pasos = 10 ** 5
    
    for i in range(pasos):
        sistema.step()
        if not i % 1000:
            mu = sistema.mean()
            histo.nuevo_dato(mu)
            print(f"ETA: ----- {100.0 * i / pasos:.2f}%")
    
    return {"N": sistema.n,
            "histo": histo,
            "bar_x_t": bar_x_t,
            "sigma_bar_x_t": sigma_bar_x_t
            }

#%% Simulaciones
#----------Parámetros de la red
N = 1000 #numero total de agentes
M = 20
hijos_mean = 2
#----------
dic = simulacion(M, hijos_mean)
histo = dic['histo']
histo.plot_densidad(scale='log')