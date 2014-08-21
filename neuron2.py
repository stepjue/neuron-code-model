from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode, odeint
from scipy.linalg import solve

def main():
    # Variable declarations
    a_j = 10E-6          # Interface and nerve cell radius (m)
    d_j = 10E-9          # Nerve cell transistor seal distance (m)
    L = 40.0E-6          # Cell length (m)
    temp = 6.3           # Temperature (deg. C)
    rho_j = 150          # Extracellular environment resistivity (Ohm*m)
    c_jg = 3.0E-3        # Gate oxide capacitance per unit area (F/m^2)
    coNa = 0.491         # Extracellular Na concentration (M)
    ciNa = 0.05          # Intracellular Na concentration (M)
    coK = 0.02011        # Extracellular K concentration (M)
    ciK = 0.400          # Intracellular K concentration (M)
    gmax_Na = 1.20E3     # Max Na conductance per unit area (S/m^2)
    gmax_K = 0.360E3     # Max K conductance per unit  area (S/m^2)
    c_m = 0.04           # Membrane capacitance per unit area (F/m^2)
    R_d = 1E3            # Drain Resistance (Ohm)
    V_to = 1.5           # Drain Potential (V)
    k = 1.0E-3           # Transistor k parameter (A/V^2)
    V_dd = 10            # Transistor drain DC bias (V)
    pulse_amp = 4.0      # Drain pulse amplitude (V)
    pulse_width = 200E-6 # Drain pulse width (s)
    b = 0.02             # Relative sodium to potassium conductance
    dt = 1.0E-6          # Time step (s)
    t_max = 20.0E-3      # Maximum time (s)
    
    # Cell parameter calculations
    area_rho = math.pi * a_j**2
    area_psi = area_rho + 2*math.pi * a_j * L
    
    R_j = rho_j / (5 * math.pi * d_j)
    
    C_psi = area_psi * c_m
    C_rho = area_rho * c_m
    C_jg = c_jg * math.pi * a_j**2
    
    v_rest = goldmann(coNa, ciNa, coK, ciK, temp, b)
    V_Na = goldmann(coNa, ciNa, 0, 0, temp, 1)
    V_K = goldmann(0, 0, coK, ciK, temp, 1)
    
    m_0 = gating_final(alpha_m, beta_m, v_rest)
    h_0 = gating_final(alpha_h, beta_h, v_rest)
    n_0 = gating_final(alpha_n, beta_n, v_rest)
    
    # Initial conditions vector
    y0 = [V_dd - R_d*k*(pulse_amp - V_to)**2, v_rest, v_rest, m_0, h_0, n_0, m_0, h_0, n_0]
    
    t = np.arange(0, t_max, dt)
    
    # Invoke the solver
    y = odeint(odefun, y0, t, args = (area_psi, area_rho, \
               gmax_Na, gmax_K, V_Na, V_K, V_dd, V_to, C_psi, \
               C_rho, C_jg, R_d, R_j, k, pulse_width, pulse_amp))
               
    v_d = y[:,0]
    v_i = y[:,1]
    v_m = y[:,2]
    
    v_j = v_i - v_m
    
    # Plot the solution
    plt.plot(t, v_j)
    plt.xlabel('t')
    plt.ylabel('v_j')
    plt.show()
    plt.plot(t, v_d)
    plt.xlabel('t')
    plt.ylabel('v_d')
    plt.show()
    plt.plot(t, v_i)
    plt.xlabel('t')
    plt.ylabel('v_i')
    plt.show()
    plt.plot(t, v_m)
    plt.xlabel('t')
    plt.ylabel('v_m')
    plt.show()

    plt.plot(t, y[:,3])
    plt.plot(t, y[:,4])
    plt.plot(t, y[:,5])
    plt.xlabel('t')
    plt.ylabel('Gating Variables (Psi side)')
    plt.show()

    plt.plot(t, y[:,6])
    plt.plot(t, y[:,7])
    plt.plot(t, y[:,8])
    plt.xlabel('t')
    plt.ylabel('Gating Variables (Rho side)')
    plt.show()


## Differential Equation Function
def odefun(y, t, area_psi, area_rho, gmax_Na, gmax_K, V_Na, V_K,
           V_dd, V_to, C_psi, C_rho, C_jg, R_d, R_j, k, pw, amp):
    v_d = y[0]
    v_i = y[1]
    v_m = y[2]
    m_psi = y[3]
    h_psi = y[4]
    n_psi = y[5]
    m_rho = y[6]
    h_rho = y[7]
    n_rho = y[8]
        
    Psi_Na = area_psi * gmax_Na * (m_psi**3) * h_psi
    Psi_K = area_psi * gmax_K * n_psi**4
    Rho_Na = area_rho * gmax_Na * (m_rho**3) * h_rho
    Rho_K = area_rho * gmax_K * n_rho**4
    
    v_gs = inv_step_function(t, pw, pw, amp)
    i_d = transistor_current(k, v_gs, V_to)
    
    A = (v_d - V_dd)/R_d + i_d
    B = Rho_Na * (V_Na - v_m) + Rho_K * (V_K - v_m) + \
        Psi_Na * (V_Na - v_i) + Psi_K * (V_K - v_i)
    C = Rho_Na * (V_Na - v_m) + Rho_K * (V_K - v_m) + (v_i - v_m) / R_j
    
    solns = np.array([A, B, C])
    coeff = np.array([[-C_jg,  C_jg, -C_jg],
                     [0, C_psi, C_rho],
                     [C_jg, -C_jg, (C_jg + C_rho)]])

    dv = solve(coeff, solns)
    
    dy = [dv[0],
          dv[1],
          dv[2],
          particle_rate(alpha_m, beta_m, m_psi, v_i),
          particle_rate(alpha_h, beta_h, h_psi, v_i),
          particle_rate(alpha_n, beta_n, n_psi, v_i),
          particle_rate(alpha_m, beta_m, m_rho, v_m),
          particle_rate(alpha_h, beta_h, h_rho, v_m),
          particle_rate(alpha_n, beta_n, n_rho, v_m)]
    return dy

## Transistor Current Forcing Function
def transistor_current(k, v_gs, V_to):
    return k * (v_gs - V_to)**2 if v_gs > V_to else 0

def inv_step_function(t, delay, pw, amp):
    return amp if (t < delay or t > delay + pw) else 0

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
    return 1E3*(alpha(v)*(1 - x) - beta(v)*x)

def gating_final(alpha, beta, v):
    return alpha(v) / (alpha(v) + beta(v))

def alpha_m(v):
    if(v > 1):
        return 0
    else:
        v = 1E3 * v
        v = (v + 0.0001) if (v == -35) else v # Approximates v to prevent singularity
        a_m = -0.1 * (v + 35) / (math.exp(-0.1 * (v + 35)) - 1)
        return a_m

def alpha_h(v):
    v = 1E3 * v
    a_h = 0.07 * math.exp(-0.05 * (v + 60))
    return a_h

def alpha_n(v):
    v = 1E3 * v
    v = (v + 0.0001) if (v == -50) else v # Approximates v to prevent singularity
    a_n = -0.01 * (v + 50) / (math.exp(-0.1 * (v + 50)) - 1)
    return a_n

def beta_m(v):
    v = 1E3 * v
    b_m = 4.0 * math.exp(-(v + 60) / 18)
    return b_m

def beta_h(v):
    v = 1E3 * v
    b_h = 1 / (1 + math.exp(-0.1 * (v + 30)))
    return b_h

def beta_n(v):
    v = 1E3 * v
    b_n = 0.125 * math.exp(-0.0125 * (v + 60))
    return b_n

if __name__ == '__main__':
    main()