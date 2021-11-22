#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sistema_de_agentes import Replicadores_emparentados
from matplotlib import pyplot as plt
from medidores import Media, Histo_Log, Histo_Lineal

def simulacion(N_total, mean_hijes, pasos = 10**5,
               q=0.1, lamda=1.0, u_adim=1e-2, a=1.0, dt=1e-3,
               k_list=range(1,8)):
    """Simulación de un sistema de N_total replicadores emparentados.
    
    Devuelve un diccionario con el sistema, un histograma de la media del
    sistema, un histograma con recursos visitados por todos los agentes, 
    un histograma con los recursos segregados por número de vecinos, y
    una lista con el promedio temporal acumulado de cada agente.
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

    return {
            "sistema": sistema,
            "histo_media": histo_media,
            "histo_recursos": histo_recursos,
            "histos_vecinos": histo_vecinos,
            "dic_medias": dic_medias
            }

def media_vs_k(dic_medias):
    """ Graficar promedio de los agentes en función de su numero de vecinos"""
    for i in dic_medias:
        plt.plot(i, dic_medias[i], color='blue', ls='none', marker='.')
    plt.xlabel("Número de vecinos")
    plt.ylabel("Promedio temporal de recursos")
    
#%% Simulaciones
#----------Parámetros de la red
N_total = 10 #numero total de agentes
hijos_mean = 2
#----------

####  Simulation of the system
resultados = simulacion(N_total, hijos_mean, a=2, pasos=10**5)


#%% Gather results
histo_promedio_global = resultados['histo_media']
histo_recursos_individuales = resultados['histo_recursos']
histos_vecinos = resultados["histos_vecinos"]



#%% Visualization of results

# muestro la red con recursos individuales proporcionales al tamaño del nodo
resultados["sistema"].show_net(resultados["dic_medias"])

# histograma de todos los agentes
plt.figure()
histo_recursos_individuales.plot_densidad(scale='log')
plt.xlabel(r"$x_i$", fontsize=12)
plt.ylabel("Densidad de probabilidad", fontsize=12)
plt.tight_layout()

# histograma para cada número de conexión
plt.figure()
for histo in resultados["histos_vecinos"].values():
    histo.plot_densidad(scale='log')
plt.xlabel(r"$x_i$", fontsize=12)
plt.ylabel("Densidad de probabilidad", fontsize=12)
plt.legend([f"$k={i:d}$" for i in resultados["histos_vecinos"]], loc='best')
plt.tight_layout()

# histo_media.plot_densidad(scale='log') este capaz es mucha info

# promedio de recursos en función del número de vecinos
plt.figure()

promedios = [histos_vecinos[k].promedio() 
             for k in histos_vecinos if histos_vecinos[k].paso > 0]

k = [k for k in histos_vecinos if histos_vecinos[k].paso > 0]

plt.plot(k, promedios, marker='.', lw=1)
plt.xlabel("Número de vecinos", fontsize=12)
plt.ylabel("Riqueza promedio", fontsize=12)
plt.tight_layout()

plt.show()
