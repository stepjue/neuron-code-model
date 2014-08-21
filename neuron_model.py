# -*- coding: iso-8859-1 -*-
# neuron.py
# Neuron model
# Copyright 2014 Stephen Jue

# This file is an addition to the ahkab simulator.
"""
This module contains a neuron element and its model class.

.. image:: images/elem/neuron.svg

"""

from __future__ import division
import numpy as np
from math import exp, log, pi
from scipy.integrate import ode, odeint
from scipy.linalg import solve

from . import constants
from . import printing
from . import utilities
from . import options

A_DEFAULT = 10E-6
D_DEFAULT = 10E-9
TEMP_DEFAULT = 6.3
RHOJ_DEFAULT = 150
CJG_DEFAULT = 3.0E-3
CONA_DEFAULT = 0.491
CINA_DEFAULT = 0.05
COK_DEFAULT = 0.02011
CIK_DEFAULT = 0.400
GMAXNA_DEFAULT = 1.20E3
GMAXK_DEFAULT = 0.360E3
CM_DEFAULT = 0.04
RD_DEFAULT = 1.0E3
VTO_DEFAULT = 1.5
B_DEFAULT = 0.02
DT_DEFAULT = 1.0E-6
TMAX_DEFAULT = 20.0E-3

class neuron_model:
   def __init__(self, name, A=None, D=None, TEMP=None, RHOJ=None, CJG=None,
                CONA=None, CINA=None, COK=None, CIK=None, GMAXNA=None, GMAXK=None,
                CM=None, RD=None, VTO=None, B=None, DT=None, TMAX=None):
      self.name = name
      self.A = float(A) if A is not None else A_DEFAULT
      self.D = float(D) if D is not None else D_DEFAULT
      self.TEMP = utilities.Celsius2Kelvin(
         float(TEMP)) if TEMP is not None else TEMP_DEFAULT
      self.RHOJ = float(RHOJ) if RHOJ is not None else RHOJ_DEFAULT
      self.CONA = float(CONA) if CONA is not None else CONA_DEFAULT
      self.CINA = float(CINA) if CINA is not None else CINA_DEFAULT
      self.COK = float(COK) if COK is not None else COK_DEFAULT
      self.CIK = float(CIK) if CIK is not None else CIK_DEFAULT
      self.GMAXNA = float(GMAXNA) if GMAXNA is not None else GMAXNA_DEFAULT
      self.GMAXK = float(GMAXK) if GMAXK is not None else GMAXK_DEFAULT
      self.CM = float(CM) if CM is not None else CM_DEFAULT
      self.RD = float(RD) if RD is not None else RD_DEFAULT
      self.VTO = float(VTO) if VTO is not None else VTO_DEFAULT
      self.B = float(B) if B is not None else B_DEFAULT
      self.DT = float(DT) if DT is not None else DT_DEFAULT
      self.TMAX = float(TMAX) if TMAX is not None else TMAX_DEFAULT