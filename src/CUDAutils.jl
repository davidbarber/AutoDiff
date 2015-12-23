if PROC=="GPU"
    
    import Base.sum!
    function sum!(A::CudaArray,ind::Int,beta::Float64,out::CudaArray)
        if ind==1
            tmp=CudaArray(Float64,(1,size(A,1))); fill!(tmp,1.0)
            CUBLAS.gemm!('N','N',1.0,tmp,A,beta,out)
            free(tmp)
        elseif ind==2
            tmp=CudaArray(Float64,(size(A,2),1)); fill!(tmp,1.0)
            CUBLAS.gemm!('N','N',1.0,A,tmp,beta,out)
            free(tmp)
        else
            error("sum!(A::CudaArray,ind,beta,out) only defined for ind=1 or 2")
        end
    end
    export sum!

    
    function ArrayToCudaArray!(nodes,net)
        nds=union(nodes)
        for n in nds
            net.value[n]=CudaArray(net.value[n])
        end
    end
    
    export ArrayToCudaArray!
    
    scalval(x::CudaArray)=sqrt(CUBLAS.dot(x,x))
    export scalval

    import CUDArt.to_host
    to_host(A::Array)=A
    export to_host

    import Base.println
    println(A::CudaArray)=println(to_host(A))
    export println
end

function cArray(eltype,sz::Tuple)
    global PROC
    if PROC=="GPU"
        return CudaArray(eltype,sz)
    else
        return zeros(eltype,sz)
    end
end

function cArray(sz::Tuple)
    global PROC
    if PROC=="GPU"
        return CudaArray(Float64,sz)
    else
        return Array(Float64,sz)
    end
end

function cArray(A::Array)
    global PROC
    if PROC=="GPU"
        return CudaArray(A)
    else
        return A
    end
end

function cArray(proc::ASCIIString,A::Array)
    if proc=="GPU"
        return CudaArray(A)
    else
        return A
    end
end


export cArray


function cArray(gpu,eltype,sz::Tuple)
    if gpu==true
       return CudaArray(eltype,sz)
    else
        return zeros(eltype,sz)
    end
end



function cArray(gpu,sz::Tuple)
    if gpu==true
        return CudaArray(Float64,sz)
    else
        return Array(Float64,sz)
    end
end
function cArray(gpu,A::Array)
    if gpu==true
        return CudaArray(A)
    else
        return A
    end
end
export cArray



import Base.zeros
@gpu zeros(A::CudaArray)=CudaArray(zeros(size(A)))
zeros(A::Float64)=collect(0.0)
export zeros
