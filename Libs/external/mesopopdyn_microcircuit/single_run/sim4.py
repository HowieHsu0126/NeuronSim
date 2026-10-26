import numpy as np
from math import exp, expm1, trunc
from numpy.random import binomial
from numba import njit


def simulate(T, dt, dt_rec, params, Nrecord, seed):
    M = params["M"]
    Me = params["Me"]
    Mi = M - Me
    N = params["N"]
    
    mu = params["mu"]
    Delta_u = params["hazard_Delta_u"]
    c = params["hazard_c"]
    vreset = params["vreset"]
    vth = params["vth"]
    tref = params["tref"]
    delay = params["delay"]
    taum_e = params["taum_e"]
    taum_i = params["taum_i"]
    taus_e = params["taus_e"]
    taus_i = params["taus_i"]
    w = params["weights"]
    
    return sim(T, dt, dt_rec, M, Me, N, mu, Delta_u, c, vreset, vth, tref, delay, taum_e, taum_i, taus_e, taus_i, w, params['Iext'], Nrecord, seed)


@njit
def hazard(u, c, Delta_u, Vth):
    return c*exp((u-Vth)/Delta_u)             # hazard rate

@njit
def Pfire(u, c, Delta_u, Vth, lambda_old, dt):
    lam = hazard(u, c, Delta_u, Vth)
    Plam = 0.5 * (lambda_old + lam) * dt
    if (Plam>0.01):
        Plam = -expm1(-Plam)
    return Plam, lam



@njit
def sim(T, dt, dt_rec, M, Me, N, mu, Delta_u, c, vreset, vth, tref, delay, taum_e, taum_i, taus_e, taus_i, weights, Iext, Nrecord, seed):

    np.random.seed(seed)

    Mi = M - Me
    n_ref = round(tref/dt)
    n_delay = round(delay/dt)
    
    #membrane time constants
    tau = np.zeros(M, dtype=np.float32)
    dtau = np.zeros(M, dtype=np.float32)
    
    tau[:Me] = taum_e
    tau[Me:M] = taum_i
    dtau[:Me] = dt/taum_e
    dtau[Me:M] = dt/taum_i

    #synaptic time constants
    Etaus = np.zeros(M, dtype=np.float32)
    for m in range(Me):
        if taus_e > 0:
            Etaus[m] = exp(-dt / taus_e)
    for m in range(Mi):
        if taus_i > 0:
            Etaus[Me+m] = exp(-dt / taus_i)

    #quantities to be recorded
    Nsteps = round(T/dt)
    Nsteps_rec = round(T/dt_rec)
    Nbin = round(dt_rec/dt) #bin size for recording 
    Abar = np.zeros((Nrecord, Nsteps_rec))
    A = np.zeros((Nrecord, Nsteps_rec), dtype=np.float32)

    #initialization
    L = np.zeros(M, dtype=np.int64)
    for i in range(M):
        L[i] = round((5 * tau[i] + tref) / dt) + 1 #history length of population i
    Lmax = np.max(L)
    S = np.ones((M, Lmax))
    u=vreset * np.ones((M, Lmax), dtype=np.float32)
    n = np.zeros((M, Lmax))
    lam = np.zeros((M, Lmax))
    x = np.zeros(M)
    y = np.zeros(M)
    z = np.zeros(M)
    for i in range(M):
        n[i,L[i]-1] = 1. #all units fired synchronously at t=0
        
    h = vreset * np.ones(M)
    lambdafree = np.zeros(M)
    for i in range(M):
        lambdafree[i]=hazard(h[i], c, Delta_u[i], vth)

    #begin main simulation loop
    for ti in range(Nsteps):
        t = dt*ti
        i_rec = trunc(ti/Nbin)

#        synInput = np.zeros(M)
        synInput = weights @ y
        extInput = Iext[i_rec]
        
        for i in range(M):
            x[i] += S[i,0] * n[i, 0]
            z[i] += (1 - S[i,0]) *  S[i,0] * n[i, 0]
            h[i] += dtau[i]*(mu[i] + extInput[i] - h[i]) + synInput[i] * dt
            Plam, lambdafree[i] = Pfire(h[i], c, Delta_u[i], vth, lambdafree[i], dt)
            W = Plam * x[i]
            X = x[i]
            Z = z[i]
            Y = Plam * z[i]
            z[i] = (1-Plam)**2 * z[i] + W
            x[i] -= W
            
            for l in range(1,L[i]-n_ref):
                u[i, l-1] = u[i,l] + dtau[i] * (mu[i] + extInput[i] - u[i,l])  + synInput[i] * dt
                Plam, lam[i,l-1] = Pfire(u[i,l-1], c, Delta_u[i], vth, lam[i,l], dt)
                m = S[i, l] * n[i,l]
                v = (1 - S[i,l]) * m
                W += Plam * m
                X += m
                Y += Plam * v
                Z += v
                S[i,l-1] = (1 - Plam) * S[i, l]
                n[i,l-1] = n[i,l]
            for l in range(L[i]-n_ref,L[i]):  #refractory period
                X+=n[i,l]
                n[i,l-1]=n[i,l]
   
            if (Z>0):
                PLAM = Y/Z
            else:
                PLAM = 0
            
            nmean = max(0, W +PLAM * (1 - X))
            if nmean>1:
                nmean = 1

            n[i, L[i]-1] = binomial(N[i], nmean) / N[i] # population activity (fraction of neurons spiking)
            
            y[i] = y[i] * Etaus[i] + n[i, L[i] - n_delay-1] / dt * (1 - Etaus[i])

            if (i <= Nrecord):
                Abar[i, i_rec] += nmean
                A[i,i_rec] += n[i,L[i]-1]
                
        # if np.mod(ti+1,Nsteps/100) == 0:  #print percent complete
        #     print("\r%d%% "%(np.round(100*ti/Nsteps),))

        
    Abar /= (Nbin * dt)
    A  /= (Nbin * dt)


    

    print("\r")
    
    return Abar, A
