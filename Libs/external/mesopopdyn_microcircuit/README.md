# Mesoscopic (and microscopic) simulation of cortical microcircuit

This software simulates a cortical microcircuit based on the Potjans-Diesmann model. It is implemented in different languages, Python, Julia, C and NEST. The C code represents an improved implementation of the original simulation code in the repository [mesopopdyn_gif](https://github.com/schwalger/mesopopdyn_gif).

There are the following implementations:
- Microscopic simulation (my custom C code)
- Microscopic simulation (NEST)
- Mesoscopic simulation (my custom C code)
- Mesoscopic simulation (NEST)
- Mesoscopic simulation (Python with Numba)
- Mesoscopic simulation (Python without Numba)
- Mesoscopic simulation (Julia)

For the C and NEST simulations you need to run PDmodel_C-code-tilo_NEST.py. For the Python simulations you need to run PDmodel_mesopython.py. For the Julia simulation you need to run PDmodel_mesojulia.jl

There are three folders:

1. single_run/: simulations of a single trial of the PD model as in Fig.1 and Fig. 8 of Schwalger et al. PLoS Comput Biol (2017) but without adaptation

2. benchmarking/: simulations of 10s of biological time to measure execution times

3. mesopopdyngif_C_v18_py3/: contains the C code for micro- and mesoscopic simulations as well as the python modules that provide a Python interface. You need to build the C libraries with make (and ignore warnings).
