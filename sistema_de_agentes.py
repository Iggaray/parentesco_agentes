#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:03:15 2021

@author: nacho
"""

from numpy.random import default_rng
import numpy as np
from kinship_network import Kinship_net

class Sistema(object):
    """Sistema de agentes con recursos"""
    
    def __init__(self, n, x0, q, lamda, u, dt):
        self.n = n #numero de agentes
        self.x = x0 #recursos iniciales de los agentes
        self.q = q
        self.p_res = q * dt #probabilidad de reseteo en dt
        self.rng = default_rng(123456) #generador aleatorio
        self.lamda = lamda
        self.mu = lamda * dt #paso de crecimiento mult.
        self.u = u
        self.dt = dt
        self.t = 0
    
    def step(self):
        """Ejecutar un paso de evolución"""
        pass
    
    def transitorio(self, pasos):
        """Ejecutar un número de pasos de evolución"""
        for i in range(pasos):
            self.step()
    
    def mean(self):
        """Calcular el promedio de recursos del sistema"""
        return self.x.mean()
    
    def std(self):
        """Calcular el desvío estándar de recursos del sistema"""
        return self.x.std

class Replicadores_emparentados(Sistema):
    """Sistema de agentes replicadores-cooperativos con reseteo estocástico.
    
    La cooperación de este sistema está dada por un grafo no dirigido que
    representa vínculos de parentesco entre agentes. Cada agente sigue una
    ecuación dinámica del tipo
    
        \dot{x_i} = x_i * (\lambda_i - \sum_j^N \lambda_j*x_j) +
                    a_i * (\hat x - x_i) +
                    (u_i - x_i)\delta(t - t_ik),
                    
    donde \hat x representa el promedio sobre todos los primeros vecinos de
    un agente en el grafo de parentescos.
    
    Parameters
        ----------
        M : int
            Número de parejas iniciales.
        n_mean : float
            Número medio de hijes por pareja.
        x0 : float
            Recursos iniciales de cada nodo.
        q : float
            tasa de reseteo de cada nodo.
        lamda : float
            tasa de crecimiento de cada nodo.
        u : float
            valor de reseteo de cada nodo.
        a : float
            tasa de coparticipación de recursos a primeros vecinos.
        dt : float
            paso temporal de integración.
    """
    
    def __init__(self, N_total, n_mean, x0, q, lamda, u, a, dt):
        
        
        self.red = Kinship_net(N_total, n_mean)
        
        self.vecinos = self.red.generar_dic_vecinos()
        
        
        super().__init__(
            N_total,
            x0 * np.ones(N_total),
            q * np.ones(N_total),
            lamda * np.ones(N_total),
            u * np.ones(N_total) / N_total,
            dt
            )
        self.a = np.ones(N_total) * a * dt
    
    
    
    def step(self):
        """Paso de evolución """
        
        # calculo fitness promedio
        bar_mu = (self.mu * self.x).sum()
        
        # calculo el promedio de los vecinos de cada agente x_i
        hat_x = [self.x[self.vecinos[i]].mean() for i in range(self.x.size)]
        hat_x = np.array(hat_x)
        
        # aplico el paso determinista de evolución
        self.x += self.x * (self.mu - bar_mu) + self.a * (hat_x - self.x)
        
        # aplico los reseteos estocásticos
        reseteos = self.rng.random(size=self.n) < self.p_res
        self.x[reseteos] = self.u[reseteos]
        
        self.t += self.dt
    
    def show_net(self, dic_recursos):
        """Mostrar grafo de parentescos
        
        Muestra una red con todos los agentes como vinculados. El tamaño de
        nodo se fija según dic_recursos, diccionario con los recursos de cada
        agente.
        """
        self.red.show(dic_recursos)
