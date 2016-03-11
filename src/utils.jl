void(x)=()

#=
import Base.convert
function convert(net::network,gpucpu::ASCIIString)
    
    #if (net.gpu & (gpucpu=="GPU")) | (!net.gpu & (gpucpu=="CPU"))
    #    return net
    #else
    #netout=deepcopy(net)
    netout=deepcopy(net)
    
    nds=net.validnodes
    if gpucpu=="GPU"
        handle= cudnnCreate()
        netout.handle = handle
        netout.gpu=true
        for n in nds
            # Check tensor
            if typeof(net.node[n]) == ADTensor
            tNode = net.node[n]
            dType = cudnnDataTypeCheck(eltype(net.value[n]))
            #check filter
            if tNode.filter == false
            v = NDTensor(handle,CudaArray(net.value[n]),tNode.dims,tNode.stride,dType)
            netout.value[n] = v
            netout.gradient[n]=CudaArray(net.gradient[n])
            else
            netout.value[n] = NDFilter(CudaArray(net.value[n]),tNode.dims,dType)
            netout.gradient[n]=CudaArray(net.gradient[n])
            end
            
            else
            netout.value[n]=CudaArray(net.value[n])
            netout.gradient[n]=CudaArray(net.gradient[n])
            end
            if isdefined(net.auxvalue,n)
                if isa(net.auxvalue[n],Tuple)
                    tmp=Array(Any,length(net.auxvalue[n]))
                    for i=1:length(net.auxvalue[n]) # can be a tuple, so iterate over elements
                        tmp[i]=nothing
                        if  ~isa(net.auxvalue[n][i],Void) && ~isempty(net.auxvalue[n][i])
                            tmp[i]=CudaArray(net.auxvalue[n][i])
                        end
                    end
                    netout.auxvalue[n]=tuple(tmp...)
                else
                    if  ~isa(net.auxvalue[n],Void) && ~isempty(net.auxvalue[n])
                        netout.auxvalue[n]=CudaArray(net.auxvalue[n])
                    end
                end
            end
        end
    end
    if gpucpu=="CPU"
        netout.gpu=false
        for n in nds
            netout.value[n]=to_host(net.value[n])
            netout.gradient[n]=to_host(net.gradient[n])
            if isdefined(net.auxvalue,n)
                if  isa(net.auxvalue[n],CudaArray)
                    netout.auxvalue[n]=to_host(net.auxvalue[n])
                end
            end
        end
    end
    return netout
#end
end
export convert
=#


import Base.copy!
@gpu copy!(gB::CudaArray,gA::CudaArray)=CUBLAS.blascopy!(length(gA),gA,1,gB,1)
export copy!

@gpu axpy!(a,x::CudaArray,y::CudaArray)=CUBLAS.axpy!(length(x),a,x,1,y,1)
axpy!(a,x::Array,y::Array)=BLAS.axpy!(a,x,y)
axpy!(a::Array{Float64,1},x::Array,y::Array)=BLAS.axpy!(a[1],x,y)
export axpy!

function A_elmult_B_update!(alphascal,A,B,betascal,C)
#    broadcast!(*,C,A,B) # inplace Hadamard product updationg result in C
    m,n = size(A)
    @assert (m,n) == size(B)
    for j in 1:n
        for i in 1:m
            @inbounds C[i,j] = alphascal*A[i,j]*B[i,j]+betascal*C[i,j]
        end
    end
end
export A_elmult_B_update!


function axpy(a::Float64,x::Array,y::Array)
    out=copy(y)
    axpy!(a,x,out)
    return out
end
export axpy

#@gpu using CUDArt, CUBLAS

if PROC=="GPU"
    import Base.LinAlg.BLAS.scale!
    scale!(s,g::CudaArray)=CUBLAS.scal!(length(g),s,g,1)
    export scale!

    # CUDA modules:
    include("CUDAmodules.jl")
    include("cuda_utils/CUDA/CuDNN.jl")

end # end of GPU stuff

sigma(x)=1.0./(1.0+exp(-x))

sigmoid(x)=sigma(x)  # TODO: replace sigma with sigmoid throughout

sigma(x)=1./(1.+exp(-x))

export sigma


if PROC=="GPU"
    function extract(A::CudaArray)
    return to_host(A)
    end
end
extract(A)=A
export extract




function ensurearray(x)
    if isa(x,Vector)
        y=Array(typeof(x[1]),length(x),1)
        y[:]=x
        return y
    else
        return x
    end
end
export ensurearray

function converttype!(x,intype,outtype)
    for i=1:length(x)
        if typeof(x[i][1])==intype
            x[i]=convert(outtype,x[i])
        end
    end
end

# find the max value of each column of a matrix:
function argcolmax(x)
    out=Array(Int,size(x,2))
    for i=1:length(out)
        out[i]=indmax(x[:,i])
    end
    return out
end
export argcolmax

softmax(x::Array{Float64,2})=exp(x)./sum(exp(x),1)
export softmax

function logsumexp(x)
    xmax=maximum(x)
    return log(sum(exp(x-xmax),1))+xmax
end



p=10e-25
mylog(x)=log(x+ep)

function log1pexp(x)
    z=zeros(size(x))
    y=copy(x)
    z[x.>0]=x[x.>0]
    y[x.>0]=-x[x.>0]
    return z+log(1+exp(y))
end




isaScalar(x)=length(size(x))==0
export isaScalar




function printover(s::AbstractString, color::Symbol = :color_normal) # nicked from ProgressMeter
    print("\u1b[1G")   # go to first column
    print_with_color(color,  s)
    print("\u1b[K")    # clear the rest of the line
    end
    export printover


function numel(x)
    sum(map((x)->length(x),x))
    end
    export numel


#import Base.contains
function contains(haystack::Array{Int64,1},needle::Int64)
    # return true if haystack contains the specified needle
    return length(find(haystack.==needle))>0
end
export contains



targ(x,a)= isa(x,Tuple)? x[a] : x # tuple argument
export targ




function copyind!(dest,source,ind) # copy only specific components of source to dest
    for i in ind
        copy!(dest[i],source[i])
    end
end
export copyind!




function toposort(node::Array{ADnode,1}) # topological sort
    node2=deepcopy(node)
    idx=map(x->x.index,node)
    for i in idx
        node2[idx[i]]=node[i]
    end
    return node2
end
export toposort



# CUDArt deepcopy is buggy, hence:
function mydeepcopy(A)
    B=similar(A);
    map(i->B[i]=A[i],1:length(A))
    return B
end
export mydeepcopy