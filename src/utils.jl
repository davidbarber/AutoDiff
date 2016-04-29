void(x)=()


import Base.convert
#function convert(net::network,gpucpu::ASCIIString)
function convert(net::network,gpucpu::ASCIIString,eltype=Float64)

    #if (net.gpu & (gpucpu=="GPU")) | (!net.gpu & (gpucpu=="CPU"))
    #    return net
    #else
    #netout=deepcopy(net)
    netout=deepcopy(net)
    nds=net.validnodes
    netout.eltype=eltype
    #if gpucpu=="GPU" || gpucpu=="GPU32"
    if gpucpu=="GPU"
        netout.gpu=true
        for n in nds
            netout.value[n]=CudaArray(eltype,net.value[n])
            netout.gradient[n]=CudaArray(eltype,net.gradient[n])
            if isdefined(net.auxvalue,n)
                if isa(net.auxvalue[n],Tuple)
                    tmp=Array(Any,length(net.auxvalue[n]))
                    for i=1:length(net.auxvalue[n]) # can be a tuple, so iterate over elements
                        tmp[i]=nothing
                        if  ~isa(net.auxvalue[n][i],Void) && ~isempty(net.auxvalue[n][i])
                            tmp[i]=CudaArray(eltype,net.auxvalue[n][i])
                        end
                    end
                    netout.auxvalue[n]=tuple(tmp...)
                else
                    if  ~isa(net.auxvalue[n],Void) && ~isempty(net.auxvalue[n])
                        netout.auxvalue[n]=CudaArray(eltype,net.auxvalue[n])
                    end
                end
            end
        end
    end
    if gpucpu=="CPU"
        if eltype==Float32
            warning("CPU computations currently only defined for Float32")
        end
        netout.gpu=false
        for n in nds
            netout.value[n]=map(eltype,to_host(net.value[n]))
            netout.gradient[n]=map(eltype,to_host(net.gradient[n]))
            if isdefined(net.auxvalue,n)
                if  isa(net.auxvalue[n],CudaArray)
                    netout.auxvalue[n]=map(eltype,to_host(net.auxvalue[n]))
                end
            end
        end
    end
    return netout
#end
end
export convert



import Base.copy!
@gpu copy!(gB::CudaArray,gA::CudaArray)=CUBLAS.blascopy!(length(gA),gA,1,gB,1)
export copy!

@gpu axpy!(a::Real,x::CudaArray{Float32},y::CudaArray{Float32})=CUBLAS.axpy!(length(x),Float32(a),x,1,y,1)
@gpu axpy!(a::Real,x::CudaArray{Float64},y::CudaArray{Float64})=CUBLAS.axpy!(length(x),Float64(a),x,1,y,1)

axpy!(a,x::Array,y::Array)=BLAS.axpy!(a,x,y)
axpy!(a::Array{Real,1},x::Array,y::Array)=BLAS.axpy!(a[1],x,y)
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


function axpy(a::Real,x::Array,y::Array)
    out=copy(y)
    axpy!(a,x,out)
    return out
end
export axpy

#@gpu using CUDArt, CUBLAS

if (PROC=="GPU") || (PROC=="GPU32")
    import Base.LinAlg.BLAS.scale!
    scale!(s,g::CudaArray{Float64})=CUBLAS.scal!(length(g),Float64(s),g,1)
    scale!(s,g::CudaArray{Float32})=CUBLAS.scal!(length(g),Float32(s),g,1)
    export scale!

    # CUDA modules:
    include("CUDAmodules.jl")

end # end of GPU stuff

sigma(x)=1.0./(1.0+exp(-x))

sigmoid(x)=sigma(x)  # TODO: replace sigma with sigmoid throughout

sigma(x)=1./(1.+exp(-x))

export sigma


if (PROC=="GPU") || (PROC=="GPU32")
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

function convert!(outtype,x::Array)
    for i=1:length(x)
        if isdefined(x,i)
            x[i]=convert(outtype,x[i])
        end
    end
end
export convert!


# find the max value of each column of a matrix:
function argcolmax(x)
    out=Array(Int,size(x,2))
    for i=1:length(out)
        out[i]=indmax(x[:,i])
    end
    return out
end
export argcolmax

softmax(x::Array{Real,2})=exp(x)./sum(exp(x),1)
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


function IsScalarArray(A::ArrayOrCudaArray)
    if prod(size(A))==1
        return true
    else
        return false
    end
end
export IsScalarArray

