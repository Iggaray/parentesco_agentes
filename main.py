#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:37:14 2021

@author: nacho
"""

from sistema_de_agentes import Replicadores_emparentados
from matplotlib import pyplot as plt
from medidores import Media, Histo_Log, Histo_Lineal

def simulacion(N_total, mean_hijes, pasos = 10**5,
               q=0.1, lamda=1.0, u_adim=1e-2, a=1.0, dt=1e-3,
               k_list=range(1,8)):
    """Simulación de un sistema de N_total replicadores emparentados.
    
    Devuelve un diccionario con el tamaño del sistema, un histograma de los
    valores totales de recursos visitados, promedio temporal acumulado y
    fluctuaciones temporales.
    """
    
    sistema = Replicadores_emparentados(
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
    
    # histograma para la media de recursos del sistema
    histo_media = Histo_Lineal(
        xmin = u_adim / sistema.n,
        xmax = 1.0 / sistema.n,
        nbins = 100
        )
    
    # histograma para los recursos de todos los agentes del sistema
    histo_recursos = Histo_Log(
        xmin = u_adim / sistema.n,
        xmax = 1.0 / sistema.n,
        nbins = 100
        )
    
    # diccionario de histogramas de recursos según el número de vecinos
    histo_vecinos = {i: Histo_Lineal(
        xmin = u_adim / sistema.n,
        xmax = 1.0 / sistema.n,
        nbins = 100
        ) for i in k_list}
    
    # diccionario con listas de nodos segun numero de vecinos (key)
    dict_vecinos = sistema.red.segregate_per_neighbours(k_list)
    
    # lista con promedios acumulados de cada agente (se van a ir actualizando)
    lista_medias = [Media() for i in range(N_total)]
 
    for i in range(pasos):
        sistema.step()
        if not i % 1000:
            # agrego nuevo promedio al hsitograma
            mu = sistema.mean()
            histo_media.nuevo_dato(mu)
            
            # agrego datos de recursos de todo el sistema al histograma
            histo_recursos.nuevo_dato(sistema.x)
            for n, histo in list(
                    zip(dict_vecinos.keys(), dict_vecinos.values())):
                histo_vecinos[n].nuevo_dato(sistema.x[dict_vecinos[n]])
            
            # actualizo promedios individuales
            for j, media in enumerate(lista_medias):
                media.nuevo_dato(sistema.x[j])            
            
            print(f"Completado: ----- {100.0 * i / pasos:.2f}%")
    

    # diccionario con media acumulada de recursos de cada agente
    dic_medias = {i: media.get() for i, media in enumerate(lista_medias)}
    
    # muestro la red de replicadores emparentados con sus recursos promedio
    sistema.show_net(dic_medias)

    return {"N": sistema.n,
            "mean_hijes": mean_hijes,
            "a": a,
            "histo_media": histo_media,
            "histo_recursos": histo_recursos,
            "histos_vecinos": histo_vecinos,
            "dic_medias": dic_medias
            }

#%% Simulaciones
#----------Parámetros de la red
N_total = 10 #numero total de agentes
hijos_mean = 2
#----------

####  Simulation of the system
dic = simulacion(N_total, hijos_mean, a=1)


#Gather results
histo_media = dic['histo_media']
histo_recursos = dic['histo_recursos']

histo_recursos.plot_densidad(scale='log')
histo_media.plot_densidad(scale='log')


# Visualization of net


#recursos = np.random.rand(20)
#dic = {i: recursos[i] for i in range(0, 20)}
#red = Kinship_net(20, 2)
#red.show(dic)


### Visualization of resources


plt.figure()
plt.legend(["Recursos", "Media"], loc='best')

for histo in dic["histos_vecinos"].values():
    histo.plot_densidad(scale='log')

plt.legend([f"$k={i:d}$" for i in dic["histos_vecinos"]], loc='best')

plt.show()
