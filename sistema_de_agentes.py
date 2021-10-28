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
    """Sistema de agentes multiplicativos"""
    
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
        self.t += self.dt
    
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

class Replicadores(Sistema):
    """Sistema de agentes replicadores con reseteo estocástico.
    
    Clase que representa un sistema de agentes competitivos según el modelo
    del replicador que sufren reseteos estocásticos a tasa q:
        \dot{x_i} = x_i (\lambda_i - \sum_j^N \lambda_j*x_j) +
        (u_i - x_i)\delta(t - t_ik),
    donde x_i son los recursos del i-ésimo agente, \lambda_i su fitness, u_i
    su valor de reseteo, t_ik los instantes de reseteo del i-ésimo agente.
    
    public methods
    --------------
    step: paso de evolución
    
    """
    
    def __init__(self, n, x0, q, lamda, u, dt=1e-3):
        super().__init__(n, x0, q, lamda, u, dt)
    
    def step(self):
        """Paso de evolución """
        bar_mu = (self.mu * self.x).sum()
        self.x *= 1.0 + self.mu - bar_mu
        self.t += self.dt
        reseteos = self.rng.random(size=self.n) < self.p_res
        self.x[reseteos] = self.u[reseteos]

class Replicadores_parentesco(Sistema):
    """Sistema de agentes replicadores-cooperativos con reseteo estocástico.
    
    La cooperación de este sistema está dada por un grafo no dirigido que
    representa vínculos de parentesco entre agentes. Cada agente sigue una
    ecuación dinámica del tipo
        \dot{x_i} = x_i * (\lambda_i - \sum_j^N \lambda_j*x_j) +
                    a_i * (\hat x - x_i) +
                    (u_i - x_i)\delta(t - t_ik),
    donde \hat x representa el promedio sobre todos los primeros vecinos de
    un agente en el grafo de parentescos."""
    
    def __init__(self, M, n_mean, x0, q, lamda, u, a, dt):
        #genero la red de parentesco
        self.red = Kinship_net(M, a = 1.0 + 1.0 / n_mean)
        self.vecinos = self.red.generar_dic_vecinos()
        n = max(self.red.nodos) #numero de agentes
        super().__init__(
            n,
            x0 * np.ones(n),
            q * np.ones(n),
            lamda * np.ones(n),
            u * np.ones(n) / n,
            dt
            )
        self.a = np.ones(n) * a * dt
    
    def step(self):
        """Paso de evolución """
        #calculo fitness promedio
        bar_mu = (self.mu * self.x).sum()
        #calculo el promedio de vecindades
        hat_x = np.array([self.x[self.vecinos[i]].mean()
                          for i in range(self.x.size)])
        self.x += self.x * (self.mu - bar_mu) + self.a * (hat_x - self.x)
        self.t += self.dt
        reseteos = self.rng.random(size=self.n) < self.p_res
        self.x[reseteos] = self.u[reseteos]
    
    def show_net(self):
        self.red.show()
        
class Desacoplados(Sistema):
    """Sistema de agentes multiplicativos con reseteo estocástico."""
    def __init__(self, n, x0, q, lamda, u, dt=1e-3):
        super().__init__(n, x0, q, lamda, u, dt)
    
    def step(self):
        self.x *= 1.0 + self.mu
        self.t += self.dt
        reseteos = self.rng.random(size=self.n) < self.p_res
        self.x[reseteos] = self.u[reseteos]
        #if any(reseteos):
            #print(f"Resetearon {reseteos.sum():d} agentes en t={self.t:.3f}")
    