module AutoDiff
# (c) David Barber, University College London 2015

push!(LOAD_PATH,pwd())
#push!(LOAD_PATH, joinpath(pwd(), "src"))

# push all subdirectories from src
#map(d -> push!(LOAD_PATH, joinpath(pwd(), "src", d)),
#    filter(d -> isdir(joinpath("src", d)), readdir("src")))#
#
#push!(LOAD_PATH, joinpath(pwd(), "Demos"))

include("gpumacros.jl")
#export @gpu, @cpu

include("initfile.jl")
#include("cuda_utils/Node.jl")

@cpu println("Using CPU")
#@gpu println("Compiling kernels...")
#@gpu include("compile_kernels.jl")
@gpu println("Using GPU")

@gpu (using CUDArt; println("using CUDArt"))
@gpu (using CUBLAS; println("using CUBLAS"))

using Base.LinAlg.BLAS
using Reexport
@gpu (@reexport using CUDArt;println("reexport using CUDArt"))

global nodecounter
export nodecounter

global GPU

#@gpu ArrayOrCudaArray = Union{Array,CudaArray}
#ArrayOrCudaArray = Array




function StartCode()
    global nodecounter = 0
    global forwardNodes=[]
    global backwardNodes = []
    global params =[]
end
export StartCode

function EndCode()
    return network()
end
export EndCode

function NodeCounter()
    global nodecounter
    nodecounter
end
export NodeCounter

function getForwardNodes()
    global forwardNodes
    forwardNodes
end
export forwardNodes

function getBackwardNodes()
    global backwardNodes
    backwardNodes
end
export backwardNodes

function getParams()
    global params
    params
end


# Few future design notes: 
# ADnode and network are used to track the index dependency information 
# 1. Nodes are used to track index dependency, so that this can be simplified 
# 2. The field f, f_inplace, df can be remove:
#     i) give each operation a specific arguments state. For example, MeanState
#        or ConvolutionState which have information about operation input
#    ii) This will allow to implement more general and flexible function for forward
#        and backward proporgation:
#        forward(Opstate,netState)
#        backward(Opstate,netState)
#        allocate(Opstate)
# 3. Base on above design then additional type is need:
#    i) CPUNetState -> carry the CPU operation state
#    ii) GPUNetState -> carry the GPU operation state
# 4. Then the complie function should be slightly modified:
#    i) compile(net,"CPU") => return CPUNetState
#   ii) compile(net,"GPU") => return CPUNetState
#  iii) ADForward(state::CPUNetState) <=> ADBackward(state::CPUNetState)
#   iv) ADForward(state::GPUNetState) <=> ADBackward(state::GPUNetState)
# 5. Advantage of doing this:
#   i) Better structure, always easier for future improvement and extension 
#  ii) Structural design will help code and memory optimization like GPU memory Coalescing
# iii) Allowed more CuDNN function
# Type Hierarchy
abstract ADnode 
abstract ADValueNode <:ADnode
abstract ADdummy 
abstract ADFunctionNode <:ADnode


type ADFunction <:ADFunctionNode
index::Int
parents::Array{Int,1}
children
f::Function
f_inplace::Function
df::Function
malloc
ADFunction(f::Function,operands::ADnode...;malloc=true) = begin
        
        operands = collect(operands)
        if(isempty(operands))
        throw("function must have inputs")
        end
    
        global nodecounter+=1
        idx = nodecounter    
        #println("The $(idx)th function is $(f)")
        parents = map(n->n.index,operands) 
        #parents is need for forward pass
        children = map(n->((n!=nothing)? n.index:nothing),filter(n->isa(n,ADVariable),operands))
        # in backward pass differentiable parents become children
        if !isempty(children)
        thisnode = new(idx,parents,children,f,Inplace[f],Derivative[f],malloc)
        push!(forwardNodes,thisnode) # forward accumulation
        unshift!(backwardNodes,thisnode) #backward accumulation
        return ADVariable(idx)
        else
        thisnode = new(idx,parents,nothing,f,Inplace[f],Derivative[f],malloc)
        push!(forwardNodes,thisnode)
        return ADconst(idx)
        end
        end
end

export ADFunction


#TODO: here might be some bugs, what if user called ADVariable(idx) ?
type ADParams <:ADValueNode
index::Int
size
ADParams() = begin
                  thisnode = ADVariable()
                  push!(params,thisnode.index)
                  return thisnode
                  end

ADParams(size::NTuple) = begin
                            thisnode = ADVariable(size)
                            push!(params,thisnode.index)
                            return thisnode
                           end
end
export ADParams




type ADVariable<: ADValueNode
index::Int
size
ADVariable() = begin
                  global nodecounter+=1
                  thisnode = new(nodecounter,nothing)
                  unshift!(forwardNodes,thisnode)
                  return thisnode
                  end

ADVariable(idx::Int)=begin
                    thisnode = new(idx,nothing)
                    return thisnode
                    end

ADVariable(size::NTuple) = begin
                            global nodecounter+=1
                            thisnode = new(nodecounter,size)
                            unshift!(forwardNodes,thisnode)
                            return thisnode
                           end
end
export ADVariable

Tensor(size::NTuple{4,Int}) = ADVariable(size)
export Tensor
Filters(size::NTuple{2,Int}) = ADParams(size)
export Filters


type ADconst <:ADValueNode
index::Int
value
size
ADconst() = begin
            global nodecounter+=1
            thisnode = new(nodecounter,nothing,nothing)
            return thisnode
            end
ADconst(value::Float64) = begin
        global nodecounter+=1
        global node
        tmp=Array(Float64,(1,1))
        fill!(tmp,value)
        thisnode = new(nodecounter,tmp,nothing)
        unshift!(forwardNodes,thisnode) #push the scalar constant to the top
        return thisnode
    end
ADconst(idx::Int)= begin    
                return new(idx,nothing,nothing) 
                end


end
export ADconst

type ADtrans<: ADdummy # transpose node. Dummy node that can be used for code optimisation
    index # node index
    parent # node parent index
    input::Bool # we set this to true since this prevents dummy nodes being differentiated
    ADtrans(parentnode)=
        begin
            global nodecounter+=1
            global node
            if isa(parentnode,ADnode)
                parent=parentnode.index
            end
            thisnode=new(nodecounter,parent,true)
            if isempty(node)
                node=Array(Any,0)
            end
            push!(node,thisnode)
            return thisnode
        end
end
export ADtrans



type ADdiag<:ADdummy # Dummy diag node that can be used for code optimisation
    index # node index
    parent # node parent index
    input::Bool # we set this to true since this prevents dummy nodes being differentiated
    ADdiag(parentnode)=
        begin
            global nodecounter+=1
            global node
            if isa(parentnode,ADnode)
                parent=parentnode.index
            end
            thisnode=new(nodecounter,parent,true)
            if isempty(node)
                node=Array(Any,0)
            end
            push!(node,thisnode)
            return thisnode
        end
end
export ADdiag

type network
    forwardNodes::Array{ADnode,1}
    backwardNodes::Array{ADnode,1} # Node that forms the scalar function (by default the last node in the graph)
    params::Array{Int,1}
    value
    auxvalue
    gradient
    handle
   function network()
        return new(getForwardNodes(),getBackwardNodes(),getParams(),Array(Any,NodeCounter()),Array(Any,NodeCounter()),Array(Any,NodeCounter()),nothing)
    end
end



ArrayADnode=Array{ADnode}
ADnodeOrArrayADnode=Union{ADnode,Array{ADnode}}
export ADnodeOrArrayADnode, ArrayADnode

import Base.getindex
getindex(x::Array,A::ADnode)=getindex(x,A.index)
export getindex

import Base.setindex!
setindex!(x::Array,value,A::ADnode)=setindex!(x,value,A.index)

function setindex!(x::Array,value::Float64,A::ADnode)
    tmp=cArray((1,1))
    fill!(tmp,value)
    setindex!(x,value,A.index)
end
setindex!(x::Array,value::Float64,A::ADVariable)=setindex!(x,value,A)

function setindex!(x::Array,value,A::ADVariable)
setindex!(x,value,A.index)
end

export setindex!

include("utils.jl")
include("CUDAutils.jl")

include("defs.jl")
include("compile.jl")

include("ADforward!.jl")
include("ADbackward!.jl")

include("gradcheckGPU.jl"); export gradcheckGPU
include("gradcheckCPU.jl"); export gradcheckCPU

include("netutils.jl")
#=
function gradcheck(net;showgrad=false)
    if net.gpu
        gradcheckGPU(net;showgrad=showgrad)
    else
        gradcheckCPU(net;showgrad=showgrad)
    end
end
=#

export gradcheck

#export matread, jldopen, matopen
export ADnode, network, compile
export ADforward!, ADbackward!
#export gradcheck
#export ArrayOrCudaArray


#ADvariable(;returnderivative=true)=ADnode(;returnderivative=returnderivative)
#export ADvariable

# make the following form an array if constval is a scalar:
#ADconst(constval)=ADnode(;returnderivative=false,isconst=true,constval=Float64(constval))
#export ADconst


#export @gpu, @cpu

end


#function MapReduce(f,op,node::Array{ADnode,1})
#    # f is the function mapped onto each node
#    # op is the binary reduction
#    f(node[1])
#    for i=2:length(node)
#        oldcounter=NodeCounter()
#        f(node[i])
#        op(oldcounter, NodeCounter())
#    end
#    return Node()[end]
#end
#export MapReduce

#function endnode(node)
#    d=falses(length(node))
#    for i=1:length(d)
#        d[i]=isdefined(node,i);
#    end
#    return last(find(d))
#end
#export endnode

# Source code generation:
#include("genFcode.jl")
#include("genRcode.jl")
#export genFcode, genRcode
