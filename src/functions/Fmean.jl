# f(x)=mean(x)

function Fmean(x...)
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(x[i])
    end
    return ([tmp/length(x)],nothing)
end

function Fmean_inplace(value::Array,auxvalue,x...) # inplace
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(x[i])
    end
    fill!(value,tmp/length(x))
end


function Dmean(derivativeIDX,f_c,faux_c,grad_c,grad_n,x...)
    axpy!(grad_c[1],ones(size(grad_n))/(length(x)*length(x[derivativeIDX])),grad_n)
end


import Base.mean

if PROC=="GPU"

    function mean(A::CudaArray)
        return flatten(Float64,CUBLAS.gemv('T',1./length(A),flatten(Float64,A),CudaArray(ones(length(A)))))
    end
    export mean

    function mean!(A::CudaArray,Out::CudaArray)
         CUBLAS.gemv!('T',1./length(A),flatten(Float64,A),CudaArray(ones(length(A))),0.0,Out)
    end
    export mean!


    function Fmean(x::CudaArray...)
        tmp=CudaArray(zeros(1,1))
        for i in 1:length(x)
            axpy!(1.0/length(x),mean(x[i]),tmp)
        end
        return (tmp,nothing)
    end


    function Fmean_inplace(value::CudaArray,auxvalue,x::CudaArray...) # inplace
        fill!(value,0.0)
        for i in 1:length(x)
            axpy!(1.0/length(x),mean(x[i]),value)
        end
    end


    function Dmean(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray...)
        tmp=CudaArray(Float64,size(grad_n))
        fill!(tmp,1.0/(length(x)*length(x[derivativeIDX])))
        axpy!(grad_c,tmp,grad_n)
        free(tmp)
    end

end

Derivative[Fmean]=Dmean # Define dictionary lookup
Inplace[Fmean]=Fmean_inplace

mean(n::ADnode)=ADnode(Fmean,n)


function mean(n::ArrayADnode)
    return ADnode(Fmean,n)
end


#mean(A::ADtrans)=ADFunction(Fmean, ftranspose(node[A.parent]))
mean(A::ADtrans)=ADFunction(Fmean, node[A.parent]) # mean(A')=mean(A)

export mean
