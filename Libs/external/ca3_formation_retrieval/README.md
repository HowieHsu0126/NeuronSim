# CA3 Pattern Formation and Retrieval Simulation

This code simulates neural pattern formation and retrieval in the CA3 subregion using the CARLsim simulator. Anyone that uses this software in a publication should please cite the reference below. Reporting on this simulation software and its results are in that article.
<br><br>Kopsick, J.D., Kilgore, J.A., Adam, G.C., and Ascoli, G.A. Formation and retrieval of cell assemblies in a biologically realistic spiking neural network model of area CA3 in the mouse hippocampus. J Comput Neurosci 52, 303–321 (2024). [https://doi.org/10.1007/s10827-024-00881-3](https://doi.org/10.1007/s10827-024-00881-3).

Users of this software should please use the Hippocampome branch of the simulator [source code](https://github.com/UCI-CARL/CARLsim6/tree/feat/ca3net). Also, users should read the "use of this branch" section of the readme file for additional information [link](https://github.com/UCI-CARL/CARLsim6/tree/feat/ca3net?tab=readme-ov-file#use-of-this-branch).

Usage and development documentation for this simulation is available [here](https://hco-dev-docs.readthedocs.io/en/latest/pattern_comp_sep/overview.html).

## Initializing Projects
<br>The simulation projects for formation and retrieval are separated into different subfolders. Each sub folder has a project and build subfolder. Users should create a new project in CARLsim for each project included here that the user wants to work with. 

Code for creating a new project is [here](https://github.com/Hippocampome-Org/spatial_nav/tree/main/scripts/new_proj). Users should use init_cs6_local.sh to create the project on their local computer. init_cs6_supcomp.sh is designed to create a project on George Mason University's supercomputer. Once a new project is made then the files from this repository that match the project should be copied into the relevant project folders to replace the original files (e.g., the main file).

The files from the "project" subfolder should be copied into the new project's project folder files. The "build" folder contents should be copied into the new project's build folder files. The new project should then be ready to be rebuilt and run as a simulation.