#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:32:08 2021

@author: nacho
"""
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

#medidores.py
class Estadistico(object):
    def __init__(self, valor=0):
        self.valor = valor
        self.paso = 0
        
    def nuevo_dato(self, x):
        self.paso += 1
    
    def reset(self, valor=0):
        self.valor = valor
        self.paso = 0
        
    def get(self):
        pass

class Media(Estadistico):
    def __init__(self):
        super().__init__(valor=0)
    
    def nuevo_dato(self, x):
        self.valor *= (self.paso)/(self.paso+1)
        self.valor += x/(self.paso+1)
        self.paso += 1
    
    def get(self):
        return self.valor

class Desvio_Estandar(Estadistico):
    '''Esta clase funciona trabajando con la varianza primero, y con un getter
    se obtiene el desvio estandar'''
    def __init__(self):
        super().__init__(valor=0)
    
    def nuevo_dato(self, x):
        self.valor *= (self.paso)/(self.paso+1)
        self.valor += x**2 /(self.paso+1)
        self.paso += 1
    
    def get(self, baricentro=0):
        return np.sqrt(self.valor - baricentro**2)
    
class Histograma(object):
    '''Histograma que usa una función de indexado para actualizar los conteos'''
    def __init__(self, bins):
        self.bins = bins
        self.cuentas = np.zeros(len(bins)) #self.cuentas = Counter()
        self.paso = 0
    
    def index(self, x):
        '''Asignar un índice al valor de entrada x'''
        pass
    
    def nuevo_dato(self, x):
        '''Agregar un conteo en el bin correspondiente'''
        self.cuentas[self.index(x)] += 1
        self.paso += 1
    
    def plot_histo(self, xlabel=None, ylabel=None, title=None):
        plt.figure()
        plt.plot(self.bins, self.cuentas.values(), linestyle='none', marker='.')
    
    def reset(self):
        self.cuentas = Counter() #np.zeros(len(self.bins))
    
    def normalizado(self):
        return self.cuentas / self.cuentas.sum()
    
    def densidad(self):
        dx = self.bins[1:] - self.bins[:-1]
        return self.normalizado()[:-1] / dx
    
    def plot_densidad(self, scale='linear'):
        f = self.densidad()
        x = (self.bins[:-1] + self.bins[1:]) / 2 #Centrado en los bines
        if scale=='linear':
            plt.plot(x[f>0], f[f>0], linestyle='-', marker='.')
        if scale=='log':
            plt.loglog(x[f>0], f[f>0], linestyle='-', marker='.')
    
class Histo_Lineal(Histograma):
    def __init__(self, xmin, xmax, nbins):
        bins = np.linspace(xmin, xmax, nbins)
        super().__init__(bins)
        self.xmin = xmin
        self.xmax = xmax
        self.nbins = nbins
    
    def index(self, x):
        index = (x-self.xmin)/(self.xmax - self.xmin) * (self.nbins-1)
        return np.clip(index, 0, self.nbins - 1).astype(int)
    
class Histo_Log(Histograma):
    def __init__(self, xmin, xmax, nbins):
        razon = (xmax/xmin)**(1.0/nbins)
        bins = xmin * np.logspace(0, nbins, num=nbins, base=razon)
        super().__init__(bins)
        self.xmin = xmin
        self.xmax = xmax
        self.nbins = nbins
        self.razon = razon
    
    def index(self, x):
        index = np.log(x / self.xmin) / np.log(self.razon)
        return np.clip(index, 0, self.nbins - 1).astype(int)