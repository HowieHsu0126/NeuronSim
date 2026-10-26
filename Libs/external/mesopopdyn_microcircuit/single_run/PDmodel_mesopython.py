# simulation single realization of Potjans-Diesmann model
# using Python implementation in sim4.py
# setting as in Fig.1 and 8 of Schwalger et al. PLoS Comput Biol (2017) but without adaptation
#
# copyright Tilo Schwalger, July 2024, tilo.schwalger@bccn-berlin.de

from sim4 import simulate
import matplotlib.pyplot as plt
import numpy as np

#load parameters for potjans column
from potjans import *

scaling_factor = 0.1 # set to 1 for original network size

#simulation time
t0 = 1.0 #time for relaxation of initial transient
T = t0 + 0.15 # simulation time (seconds)

#set parameters of step stimulation
B=15.
tstart = t0 + 0.06
duration = 0.03

M = 8       # number populations
Nrecord = 8 # number populations to be recorded

dt_record = 0.001  #bin size of population activity in seconds
dt = 0.0001        #simulation time step in seconds

#parameter values taken from corrected potjans_noadap_trajec3_mdopt.py
DeltaV = 5. * np.ones(M)
#              2/3e         4e           5e           6e           2/3i         4i           5i           6i   
mu = np.array([19.14942021, 30.80455243, 29.43680143, 34.93225564, 20.36247192, 28.06868188, 29.32954448, 32.08123423])


N = np.array([ N_full['L23']['E'], N_full['L4']['E'], N_full['L5']['E'], N_full['L6']['E'],\
               N_full['L23']['I'], N_full['L4']['I'], N_full['L5']['I'], N_full['L6']['I'],\
              ])

params = {'mu': mu,
          'hazard_c': 10.0,      #Hz
          'hazard_Delta_u': DeltaV, #mV
          'vth': neuron_params['v_thresh'] - neuron_params['v_rest'],    #mV
          'vreset': neuron_params['v_reset'] - neuron_params['v_rest'],
          'taum_e': neuron_params['tau_m']*1e-3,  # (s)
          'taum_i': neuron_params['tau_m']*1e-3,  # (s)
          'delay': d_mean['E'] * 1e-3,
          'tref': neuron_params['tau_refrac'] *1e-3,
          'taus_e': neuron_params['tau_syn_E'] * 1e-3, # (s)
          'taus_i': neuron_params['tau_syn_I'] * 1e-3, # (s)
          'M': M,
          'Me': 4,
          'N': N}



W = weight_matrix

# annealed average of quenched connectivity
W_av = W * conn_probs

# matrices W based on a different ordering of excitatory and inhibitory populations
 
#re-order columns
Js_tmp = np.zeros((M, M))
Js_tmp[:,0] = W_av[:,0] # from 2/3e
Js_tmp[:,1] = W_av[:,2] # from 4e
Js_tmp[:,2] = W_av[:,4] # from 5e
Js_tmp[:,3] = W_av[:,6] # from 6e
Js_tmp[:,4] = W_av[:,1] # from 2/3i
Js_tmp[:,5] = W_av[:,3] # from 4i
Js_tmp[:,6] = W_av[:,5] # from 5i
Js_tmp[:,7] = W_av[:,7] # from 6i

# re-order rows
Js=np.zeros((M, M))
Js[0,:]=Js_tmp[0,:] # to 2/3e
Js[1,:]=Js_tmp[2,:] # to 4e
Js[2,:]=Js_tmp[4,:] # to 4e
Js[3,:]=Js_tmp[6,:] # to 4e
Js[4,:]=Js_tmp[1,:] # to 4e
Js[5,:]=Js_tmp[3,:] # to 4e
Js[6,:]=Js_tmp[5,:] # to 4e
Js[7,:]=Js_tmp[7,:] # to 4e

# turn psc amplitudes into psp amplitudes
Js[:,:4] *= (neuron_params['tau_syn_E'] * 1e-3) / neuron_params['cm'] * 1e3
Js[:,4:] *= (neuron_params['tau_syn_I'] * 1e-3)/ neuron_params['cm'] * 1e3

#coupling strength w= Js * N
w = np.zeros((M, M))
for j in range(M):
    w[:,j] = Js[:,j] * N[j]
params['weights'] = w

#rescale network size as in microscopic simulation (../c_sim/)
N = np.round(N * scaling_factor).astype(int)  #decrease network size by scaling_factor
params['N'] = N

Nbin = round(T/dt_record)
tt= dt_record * np.arange(Nbin)

# creating step stimulus
step=np.zeros(M)
step[1] = tha_drive_vec[2]  # input to 4e
step[5] = tha_drive_vec[3]  # input to 4i
step[3] = tha_drive_vec[6]  # input to 6e
step[7] = tha_drive_vec[7]  # input to 6i

step = step / step.max() * B
Iext=np.zeros((Nbin,M), dtype=np.float32)   # external current R*Iext in (mV)
kstep1=int(tstart/dt_record)
kstep2=min(int((tstart + duration) / dt_record), Nbin)
assert kstep1 <= kstep2
for m in range(M):
    Iext[kstep1:kstep2,m] = step[m]
params['Iext'] = Iext

seed=12345          #seed for finite-size noise
rate, A = simulate(T, dt, dt_record, params, Nrecord, seed)

i0 = int(t0/dt_record)
plt.figure(14)
plt.clf()
for m in range(M):
    ax=plt.subplot(4,2,m+1)
    if m%2==0:               #excitatory populations
        newindx=int(m/2)
        plt.ylabel("rate [Hz]")
    else:                    #inhibitory populations
        newindx = 4 + int(m/2)
    
    plt.plot(tt[i0:]-tt[i0],A[newindx,i0:], color='0.7')
    plt.plot(tt[i0:]-tt[i0],rate[newindx,i0:])
    plt.ylim(ymin=0)

plt.subplot(4,2,7)
plt.xlabel("time [s]")
plt.subplot(4,2,8)
plt.xlabel("time [s]")

plt.subplots_adjust(top=0.995, bottom=0.1,left=0.12,right=0.97,hspace=0.32,wspace=0.26)
plt.show()

