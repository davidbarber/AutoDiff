# f(x)=mean(abs(x))

#TODO sum(abs)

function FmeanAbs(x...)
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(abs(x[i]))
    end
    return ([tmp/length(x)],nothing)
end

function FmeanAbs_inplace(handle,value::Array,auxvalue,x...) # inplace
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(abs(x[i]))
    end
    fill!(value,tmp/length(x))
end


function DmeanAbs(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x...)
    axpy!(grad_c[1]/(length(x)*length(x[derivativeIDX])),sign(x[derivativeIDX]),grad_n)

end

if PROC=="GPU"

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


    function FmeanAbs_inplace(handle,value::CudaArray,auxvalue,x::CudaArray...) # inplace
        fill!(value,0.0)
        for i in 1:length(x)
            axpy!(1.0/length(x),meanAbs(x[i]),value)
        end
    end


    function DmeanAbs(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray...)
        tmp=CudaArray(Float64,size(grad_n))
        vsign!(x[derivativeIDX],tmp)
        alphaaxpy!(1.0/(length(x)*length(x[derivativeIDX])),grad_c,tmp,grad_n)
        free(tmp)
    end

end

Derivative[FmeanAbs]=DmeanAbs # Define dictionary lookup
Inplace[FmeanAbs]=FmeanAbs_inplace

meanAbs(n::ADnode)=ADFunction(FmeanAbs,n)

function meanAbs(n::ArrayADnode)
    return ADFunction(FmeanAbs,n)
end


#mean(A::ADtrans)=ADFunction(Fmean, ftranspose(node[A.parent]))
meanAbs(A::ADtrans)=ADFunction(FmeanAbs, node[A.parent]) # mean(A')=mean(A)

export meanAbs
