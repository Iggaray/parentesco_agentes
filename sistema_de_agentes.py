#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 14:03:15 2021

@author: nacho
"""

from numpy.random import default_rng
import numpy as np

class Sistema(object):
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
        self.t += self.dt
    
    def transitorio(self, pasos):
        for i in range(pasos):
            self.step()
    
    def mean(self):
        return self.x.mean()
    
    def std(self):
        return self.x.std

class Replicadores(Sistema):
    def __init__(self, n, x0, q, lamda, u, dt=1e-3):
        super().__init__(n, x0, q, lamda, u, dt)
    
    def step(self):
        bar_mu = (self.mu * self.x).sum()
        self.x *= 1.0 + self.mu - bar_mu
        self.t += self.dt
        reseteos = self.rng.random(size=self.n) < self.p_res
        self.x[reseteos] = self.u[reseteos]

class Desacoplados(Sistema):
    def __init__(self, n, x0, q, lamda, u, dt=1e-3):
        super().__init__(n, x0, q, lamda, u, dt)
    
    def step(self):
        self.x *= 1.0 + self.mu
        self.t += self.dt
        reseteos = self.rng.random(size=self.n) < self.p_res
        self.x[reseteos] = self.u[reseteos]
        #if any(reseteos):
            #print(f"Resetearon {reseteos.sum():d} agentes en t={self.t:.3f}")
    