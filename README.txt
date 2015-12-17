This is some basic Julia code for AutoDiff.

This code was developed for Julia version 0.4.0


If you want to use the GPU functionality, you'll need to first compile the kernels using


julia> cd("src directory  -- it is a subdir of this README file")
julia> include("compile_kernels.jl")


This only needs to be done once. I've only tested this on a Jetson TK1 and you may need to modify the nvcc options in the file compile_kernels.jl for your architecture.



To run the demos (see the Demos folder):
start julia and then from within julia type:


julia> cd("src directory  -- it is a subdir of this README file")
julia> include("setup.jl")
julia> usegpu(false) # or usegpu(true) to make GPU functionality available
julia> using AutoDiff
julia> cd("../Demos")
julia> include("DemoMNIST.jl")

and similarly for the other demos.
