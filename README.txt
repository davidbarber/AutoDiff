This is some basic Julia code for AutoDiff.

This code was developed for Julia version 0.4.0

If you want to use NVIDIA GPU functionality, you'll need to first compile the kernels using

julia> cd("cuda_kernels directory  -- it is a subsubdir of this README file")
julia> include("compile_kernels.jl")

This only needs to be done once. I've only tested this on a Jetson TK1 and Titan GTX under ubuntu 14.04. You may need to modify the nvcc options in the file compile_kernels.jl for your architecture.


To run the demos (see the Demos folder):
start julia and then from within julia type:

julia> include("CPUstart.jl") # for CPU 

or, if you have an NVIDIA GPU 

julia> include("GPUstart.jl") # for GPU		

then

julia> cd("Demos")
julia> include("DemoMNIST.jl")
