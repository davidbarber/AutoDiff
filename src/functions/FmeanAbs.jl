# f(x)=mean(abs(x))

function FmeanAbs(x...)
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(abs(x[i]))
    end
    return ([tmp/length(x)],nothing)
end

function FmeanAbs_inplace(value::Array,auxvalue,x...) # inplace
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(abs(x[i]))
    end
    fill!(value,tmp/length(x))
end


function DmeanAbs(derivativeIDX,f_c,faux_c,grad_c,grad_n,x...)
    axpy!(grad_c[1]/(length(x)*length(x[derivativeIDX])),sign(x[derivativeIDX]),grad_n)

end

if GPU

    function meanAbs(A::CudaArray)
        return CudaArray(CUBLAS.asum(A)/length(A)*ones(1,1))
    end
    export meanAbs

    function meanAbs!(A::CudaArray,Out::CudaArray)
        copy!(Out,meanAbs(A))
    end
    export meanAbs!


    function FmeanAbs(x::CudaArray...)
        tmp=CudaArray(zeros(1,1))
        for i in 1:length(x)
            axpy!(1.0/length(x),meanAbs(x[i]),tmp)
        end
        return (tmp,nothing)
    end


    function FmeanAbs_inplace(value::CudaArray,auxvalue,x::CudaArray...) # inplace
        fill!(value,0.0)
        for i in 1:length(x)
            axpy!(1.0/length(x),meanAbs(x[i]),value)
        end
    end


    function DmeanAbs(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray...)
        tmp=CudaArray(Float64,size(grad_n))
        vsign!(x[derivativeIDX],tmp)
        alphaaxpy!(1.0/(length(x)*length(x[derivativeIDX])),grad_c,tmp,grad_n)
        free(tmp)
    end

end

Derivative[FmeanAbs]=DmeanAbs # Define dictionary lookup
Inplace[FmeanAbs]=FmeanAbs_inplace

meanAbs(n::ADnode)=ADnode(FmeanAbs,n)

function meanAbs(n::ArrayADnode)
    return ADnode(FmeanAbs,n)
end


#mean(A::ADtrans)=ADnode(Fmean, ftranspose(node[A.parent]))
meanAbs(A::ADtrans)=ADnode(FmeanAbs, node[A.parent]) # mean(A')=mean(A)

export meanAbs
