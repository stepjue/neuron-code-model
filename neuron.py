import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode, odeint

def main():
    # Variable declarations
    a = 20E-6            # Cell radius (m)
    d = 2 * a            # Cell diameter (m)
    L = 40.0E-6          # Cell length (m)
    temp = 6.3           # Temperature (deg. C)    
    gmax_Na = 1.20E3     # Max Na conductance per unit area (S/m**2)
    gmax_K = 0.360E3     # Max K conductance per unit area (S/m**2)
    coNa = 491E-3        # Extracellular Na concentration (M/L)
    ciNa = 50E-3         # Intracellular Na concentration (M/L)
    coK = 20.11E-3       # Extracellular K concentration (M/L)
    ciK = 400E-3         # Intracellular K concentration (M/L)
    c_m = 1.0E-2         # Membrane capacitance per unit area (F/m**2)
    b = 0.02             # Relative sodium to potassium conductance
    dt = 1.0E-6          # Time step (s)
    t_max = 20.0E-3      # Maximum time (s)
    pulse_width = 10E-6  # Stimulus pulse width (s)
    pulse_amp = 53E-9    # Stimulus pulse amplitude (A)
    
    # Cell parameter calculations
    area = 2 * (math.pi * a**2) + 2 * math.pi * a * L
    c = area * c_m
    maxcond_Na = area * gmax_Na
    maxcond_K = area * gmax_K
    
    v_rest = goldmann(coNa, ciNa, coK, ciK, temp, b)
    v_Na = goldmann(coNa, ciNa, 0, 0, temp, 1)
    v_K = goldmann(0, 0, coK, ciK, temp, 1)
    
    m_0 = gating_final(alpha_m, beta_m, v_rest)
    h_0 = gating_final(alpha_h, beta_h, v_rest)
    n_0 = gating_final(alpha_n, beta_n, v_rest)
    
    y0 = [v_rest, m_0, h_0, n_0] # Initial conditions vector
    t = np.arange(0, t_max, dt)

    y = odeint(f, y0, t, args=(c_m, gmax_Na, gmax_K, v_Na, v_K, pulse_width, pulse_amp))
    print(y)
    plt.plot(t, y[:,0])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()

# Differential Equation Function
def f(y, t, c_m, gmax_Na, gmax_K, v_Na, v_K, pw, amp):
    dydt = [-(gmax_K / c_m) * (y[3] ** 4) * (y[0] - v_K) - (gmax_Na / c_m) \
            * (y[1] ** 3) * y[2] * (y[0] - v_Na) + step_current(t, pw, amp) / c_m
            , particle_rate(alpha_m, beta_m, y[1], y[0])
            , particle_rate(alpha_h, beta_h, y[2], y[0])
            , particle_rate(alpha_n, beta_n, y[3], y[0]) ]
    return dydt

## Step Current Forcing Function
def step_current(t, pw, amp):
    return amp if (t >= 0 and t < pw) else 0

## Goldmann Equation Function
def goldmann(coNa, ciNa, coK, ciK, T, b):
    R = 8.314
    F = 9.648E4
    Z = 1
    
    T = T + 273.15
    v = ((R * T) / (Z * F)) * math.log((coK + b * coNa) / (ciK + b * ciNa))
    return v

## Gating Particle Functions
def particle_rate(alpha, beta, x, v):
    return (alpha(v) * (1 - x) - beta(v) * x)

def gating_final(alpha, beta, v):
    return alpha(v) / (alpha(v) + beta(v))

def alpha_m(v):
    v = v_to_mv(v)
    v = (v + 0.00001) if v == -35 else v # Approximates v to prevent singularity
    a_m = -0.1 * (v + 35) / (math.exp(-0.1 * (v + 35)) - 1)
    return a_m

def alpha_h(v):
    v = v_to_mv(v)
    a_h = 0.07 * math.exp(-0.05 * (v + 60))
    return a_h

def alpha_n(v):
    v = v_to_mv(v)
    v = (v + 0.00001) if v == -50 else v # Approximates v to prevent singularity
    a_n = -0.01 * (v + 50) / (math.exp(-0.1 * (v + 50)) - 1)
    return a_n

def beta_m(v):
    v = v_to_mv(v)
    b_m = 4.0 * math.exp(-(v + 60) / 18)
    #print(b_m)
    return b_m

def beta_h(v):
    v = v_to_mv(v)
    b_h = 1 / (1 + math.exp(-0.1 * (v + 30)))
    return b_h

def beta_n(v):
    v = v_to_mv(v)
    b_n = 0.125 * math.exp(-0.0125 * (v + 60))
    return b_n

def v_to_mv(v):
    return v * 1.0E3

if __name__ == '__main__':
    main()
