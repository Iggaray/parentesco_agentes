#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# medidores.py

import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

class Media(object):
    """Media acumulada que se alimenta de datos en serie, de a uno.
    
    Recuerda el número de datos que recibió."""
    
    def __init__(self, valor=0):
        self.valor = valor
        self.paso = 0
    
    def nuevo_dato(self, x):
        self.valor *= (self.paso)/(self.paso+1)
        self.valor += x/(self.paso+1)
        self.paso += 1
    
    def get(self):
        return self.valor

    
class Histograma(object):
    '''Histograma que usa una función de indexado para actualizar los conteos.
    
    Atributos:
    ----------
    bins : 1-d ndarray
        Contiene los bordes de los bines
    cuentas : 1-d array de enteros
        Contiene las cuentas realizadas en cada bin
    paso : int
        Número de muestras que lleva recolectadas el histograma
        
    Métodos
    -------
    index : método abstracto de indexado de bines propio de cada subclase.
    plot_histo : graficar el histograma usando matplotlib
    reset: reiniciar a cero los valores de los bines
    normalizado: devolver los conteos de los bines normalizados
    densidad: devolver la densidad de probabilidad asociada al historgama
        (tiene en cuenta el ancho de cada bin)
    plot_densidad: graficar densidad de probabilidad asociada al histograma'''
    
    def __init__(self, bins):
        self.bins = bins
        self.cuentas = np.zeros(len(bins)) #Counter()
        self.paso = 0
    
    def index(self, x):
        '''Asignar un índice al valor de entrada x
        '''
        pass
    
    def nuevo_dato(self, x):
        '''Agregar un conteo en el bin correspondiente
        '''
        if x is float:
            self.cuentas[self.index(x)] += 1
            self.paso += 1
        
        else:
            self.cuentas[self.index(x)] += 1
            self.paso += x.size

    def plot_histo(self, xlabel=None, ylabel=None, title=None):
        '''Graficar histograma
        '''
        plt.figure()
        plt.plot(self.bins, self.cuentas.values(), linestyle='none', marker='.')
    
    def reset(self):
        self.cuentas = Counter()
    
    def normalizado(self):
        if self.paso == 0: # si todos los bines son nulos
            return self.cuentas
        
        return self.cuentas / self.cuentas.sum()
    
    def densidad(self):
        dx = self.bins[1:] - self.bins[:-1]
        return self.normalizado()[:-1] / dx
    
    def bin_centers(self):
        x = (self.bins[1:] + self.bins[:-1]) / 2
        return x
    
    def plot_densidad(self, scale='linear'):
        f = self.densidad()
        x = self.bin_centers() # centrado en los bines
        if scale=='linear':
            plt.plot(x[f>0], f[f>0], ls='-', lw=1)
        if scale=='log':
            plt.loglog(x[f>0], f[f>0], ls='-', lw=1)
    
    def promedio(self):
        x = (self.bins[1:] + self.bins[:-1]) / 2
        probabilidad = self.normalizado()[:-1]
        return (x * probabilidad).sum()
    
class Histo_Lineal(Histograma):
    '''Histograma con bineo lineal
    '''
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
    '''Histograma con bineo logarítmico
    '''
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