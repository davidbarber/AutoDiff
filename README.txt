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

Note that include("GPUstart") does not mean that all subsequent computations will be performed on the GPU. It simply makes the GPU functionality available, if this might wish to be used. We still need to specify in the program whether we wish to use the CPU or GPU backend to perform the calculations.



Basic Usage:

Let's assume that we initialised by using include("CPUstart.jl").

The first thing to note is that in calculus, we can only take the derivative of scalar valued functions. For this reason, the AutoDiff graph must define the computation of a scalar value.

Let's say we want to calculate the derivative (with respect to X) of sum(A*X) where A is a matrix with given value, X is a matrix variable and sum() computes the sum over all elements of the matrix (resulting in a scalar for which we can then compute the derivative with respect to X).  We can do this by

julia>StartCode(); A=ADnode(); X=ADvariable(); out=sum(A*X); net=EndCode();

This defines an object net which contains a representation of the computation graph. This directed graph has 4 nodes, namely A (node index 1), X (node index 2) A*X (node index 3) and out (node index 4). 

A and X are nodes without parents (root nodes) and the remaining nodes have at least one parent.

 A     X
  \   /
   A*X
    |
   out   


julia> A.index  # gives value 1
julia> out.index # gives value 4
julia> net.FunctionNode  # gives value 4

By default, it is assumed that the final node in the graph computes the value of the function. net.FunctionNode contains the node index of the function.

There are two kinds of nodes:

ADnode(). These nodes will be treated as constant values and their gradients will not be computed.
ADvariable(). The gradients of these nodes will be computed.

Now that we've defined a graph, we need to instantiate the root nodes in order to allocate memory.

julia> net.value[A]=rand(2,2)
julia> net.value[X]=rand(2,1)

Since now A and X are defined (and they are the only nodes in the graph without parents), the value of the remaining nodes in the graph will inherit their values from their parents.

Now that we've specified values for the root nodes, we can compile the graph which will (amongst other things) allocate memory on the graph and specify the values of all remaining nodes in the graph.

julia> net=compile(net)
julia> net.value[3]
julia> net.value[out]

The two basic operations are 
ADforward!(net) : calculate the forward pass of all node values 
ADbackward!(net) : calcualtes the backward pass and all gradient values

julia> ADforward!(net) # this calculates or updates any node values
julia> ADbackward!(net) # calculates the gradient values

If we now examine

julia> net.gradient[X]

we will see that it contains the gradient of the function with respect to the variable X.

To do this on the GPU, we would do:

julia> include("GPUstart.jl")
julia> StartCode(); A=ADnode(); X=ADvariable(); out=sum(A*X); net=EndCode();
julia> net=compile(net)
julia> net.value[3]
julia> net.value[out]
julia> CUDArt.init([0])
julia> net=convert(net,"GPU")
julia> ADforward!(net)
julia> ADbackward!(net)
julia> net.gradient[X]

The final step above returns a CudaArray. To examine the value we can send to the host with

julia> to_host(net.gradient[X])

Or, if we wish, we can convert back to the CPU version

julia> net=convert(net,"CPU")

The philosophy is to keep the coding minimal in the sense that all the AutoDiff package is doing is making available ADforward and ADbackward passes on either the CPU or GPU. The rest (in terms of device management) is up to the user. For this reason, the user still needs to call CUDArt.init([0]) to initialise the device. 





Implemented Operations:

On the computation graph, all quantities are stored as Float64 Arrays or Float64 CudaArrays. 

Scalars are stored on the graph as (1,1) or (1,) Arrays. For notational convenience below, we call these ScalarArrays.


Elementary Matrix Operations:

A*B
A+B
A.+B
A./B
A-B
A.-B


These operations work only when the shapes of the matrices mean the operations are well defined.

A special case is that 

a*B

works if a is a ScalarArray and computes the value a[1,1]*B. 

a.*B also computes a[1,1]*B when a is a ScalarArray and B is an Array.

The following are also defined efficently by calling (CU)BLAS routines without creating a node that stores diagonal matrices:

diagm(A)*B
A*diagm(B)
diagm(A)*diagm(B)

diagm(A)+B
A+diagm(B)
diagm(A)+diagm(B)


Constants:

Let's say we want to do calculate z=3*x+4*y. The compiler understands constants, so this can be written as 

StartCode()
x=ADvariable()
y=ADvariable()
z=3*x+4*y
net=EndCode()

Transpose Operations:

The following routines are optimised by calling (CU)BLAS routines and perform the calculation without creating nodes that store the matrix transposes

A'*B
A*B'
A'*B'

A'+B
A+B'
A'+B'

Alternatively, 

trans(A)

computes a new node on the graph that contains the transpose of the matrix A. 

Mathematically, trans(A)+B and A'+B are equivalent, but the latter is more efficient since it doesn't create a new graph node that contains the transpose of A.
------------------------------------------------------------------------
diagm(A)

creates a digonal 

------------------------------------------------------------------------
sum(A)

This computes the sum over all array elements 
------------------------------------------------------------------------
sum([A B C])

This computes sum(A)+sum(B)+sum(C) 
------------------------------------------------------------------------
mean(A)

This computes the mean over all array elements, namely sum(A)/length(A)
------------------------------------------------------------------------
mean([A B C]

This computes the mean of the means. In this case, this would be (mean(A)+mean(B)+mean(C))/3
------------------------------------------------------------------------
sumSquare(A) (or sumSq(A))

Computes the sum of the squared elements of A
------------------------------------------------------------------------
sumSquare([A B])

Computes sumSq(A)+sumSq(B)
------------------------------------------------------------------------
meanSquare(A)

Computes the mean of the squared elements of A
------------------------------------------------------------------------
meanSquare([A B])

Computes (meanSq(A)+meanSq(B))/2
------------------------------------------------------------------------
meanSquareLoss(A,B)

For arrays A and B computes  sum_i((A[i]-B[i]).^2)/length(A)
------------------------------------------------------------------------
BinaryKullbackLeiblerLoss(A,B)

For arrays A and B (with A[i] and B[i] between 0 and 1) computes the average cross entropy loss

sum(A.*log(A./B)+(1.-A).*log((1.-A)./(1.-B)))/length(A)
------------------------------------------------------------------------
BinaryKullbackLeiblerLossXsigmoidY(A,B)

Computes

BinaryKullbackLeiblerLoss(A,sigmoid(B))

in an efficient and computatioanlly stable way.
------------------------------------------------------------------------
stanh(A)

Computes a scaled version of tanh(A), namely 2.5*tanh(A)
------------------------------------------------------------------------
sigmoid(A)

Computes 1./(1+exp(-A[i]))
------------------------------------------------------------------------
rectlin(A)

Computes max(0,A[i])
------------------------------------------------------------------------
kinklin(A)

Computes max(0.25*A[i],A[i])
------------------------------------------------------------------------
softmax(A)

Computes exp(A[i])./sum(exp(A[i]))
------------------------------------------------------------------------
KLsoftmax(A,B)

The Kullback-Leibler divergence between distributions A and softmax(B)

(sum_i A[i]*log(A[i])-A[i]*log(softmax(B)[i]))/length(A)
------------------------------------------------------------------------
exp(A)

Computes elementwise exponential exp(A[i])
------------------------------------------------------------------------
log(A)

Computes elementwise logarithm log(A[i])
------------------------------------------------------------------------
abs(A)

Computes elementwise absolute abs(A[i])
------------------------------------------------------------------------
meanAbs(A)

Computes mean(abs(A)
------------------------------------------------------------------------
meanAbs([A B])

Computes mean( [mean(abs(A)) mean(abs(B))])
------------------------------------------------------------------------
vcat(x....), e.g. vcat([A B C])

stacks matrices vertically. They must have matching horizontal dimensions.
------------------------------------------------------------------------
hcat(x....), e.g. hcat([A B C])

stacks matrices horizontally. They must have matching vertival dimensions.
------------------------------------------------------------------------
rectlinAXplusBias(A,X,bias)

Computes rectlin(A*X+bias) in an efficient way.
------------------------------------------------------------------------
stanhAXplusBias(A,X,bias)

Computes stanh(A*X+bias) in an efficient way.
------------------------------------------------------------------------
sigmoidAXplusBias(A,X,bias)

Computes sigmoid(A*X+bias) in an efficient way.
------------------------------------------------------------------------
kinklinAXplusBias(A,X,bias)

Computes kinklin(A*X+bias) in an efficient way.
------------------------------------------------------------------------
AXplusBias(A,X,bias)

Computes A*X+bias in an efficient way.
