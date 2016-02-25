FsumSq(x)=([sum(x.*x)],nothing) # must always return an array for the value
FsumSq_inplace(value,auxvalue,x)=copy!(value,sum(x.*x))

function FsumSq(x...)
    tmp=0.0
    for i in 1:length(x)
        tmp+=sum(x[i].*x[i])
    end
    return ([tmp],nothing) # must always return an array for the value
end

function FsumSq_inplace(value,auxvalue,x...) # inplace
    tmp=0.0
    for i in 1:length(x)
        tmp+=sum(x[i].*x[i])
    end
    copy!(value,tmp)
end


DsumSq(derivativeIDX,f_c,faux_c,grad_c,grad_n,x_n...)=axpy!(2.0,grad_c.*x_n[derivativeIDX],grad_n)


if PROC=="GPU"
    function FsumSq(x::CudaArray...)
        tmp=CudaArray(zeros(1))
        for i in 1:length(x)
            CUBLAS.gemv!('T',1.0,flatten(Float64,x[i]),vec(x[i]),1.0,tmp)
        end
    return (tmp,nothing) # must always return an array for the value
    end

    function FsumSq_inplace(value,auxvalue,x::CudaArray...) # inplace
        fill!(value,0.0)
        for i in 1:length(x)
            CUBLAS.gemv!('T',1.0,flatten(Float64,x[i]),vec(x[i]),1.0,value);
        end
    end

    function DsumSq(derivativeIDX,f_c,faux_c,grad_c,grad_n,x_n::CudaArray...)
        alphaaxpy!(2.0,grad_c,x_n[derivativeIDX],grad_n)
    end

end

Inplace[FsumSq]=FsumSq_inplace
Derivative[FsumSq]=DsumSq

sumSq(n::ADnode)=ADnode(FsumSq,n)

function sumSq(n::ArrayADnode)
    return ADnode(FsumSq,n)
end

#sumSq(A::ADtrans)=ADFunction(FsumSquare, ftranspose(node[A.parent]))
sumSq(A::ADtrans)=ADFunction(FsumSq, node[A.parent]) # sumsq(A')=sumsq(A)
sumSq(A::ADdiag)=sumSq(node[A.parent])

sumSquare=sumSq


export sumSquare
