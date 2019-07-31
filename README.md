# Optimal Aircraft Rerouting during Space Launches using Adaptive Spatial Discretization

This repository contains supplementary code written at Stanford and discussed in the paper titled “Optimal Aircraft Rerouting during Space Launches using Adaptive Spatial Discretization” by Rachael E. Tompa and Mykel J. Kochenderfer, in the 2018 Digital Avionics Systems Conference. 

Here you will find the code to perform the following:

Implement and solve the individual MDP solutions using parallelized value iteration and combining them to a single solution
  * src/CSLVProblem.jl
  * src/CSLV_.jl
  * src/ParallelVI_.jl
  * src/RunMDP.jl
  
Asset Files
  * assets/debris.jld
  * assets/flightPaths.jl
  
Visualize the MDP solution
  * notebooks/visualizeUtilityAndPolicy.ipynb
  
Run simulations of the airspace to analyze historic, nominal, and proposed aircraft flights
  * notebooks/runSimulations.ipynb
  * src/Simulations.jl

## Dependencies

The software is implemented entirely in Julia. For the best results, the user should use a notebook. An example notebook is provided for the reader's convenience in the example subdirectory. The following Julia packages are required for running all code and notebooks:
*	[GridInterpolations.jl](https://github.com/sisl/GridInterpolations.jl)
*	[Interact.jl](https://github.com/JuliaLang/Interact.jl)
* [Color.jl](https://github.com/JuliaGraphics/Colors.jl)
* [JLD.jl](https://github.com/JuliaIO/JLD.jl) [Note: additional dependencies]
* [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) [Note: additional dependencies]
*	[PGFPlots.jl](https://github.com/sisl/PGFPlots.jl) [Note: additional dependencies]
*	[ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl)
