
function FmeanSquare(x...)
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(x[i].*x[i])
    end
    return ([tmp/length(x)],nothing) # must always return an array for the value
end

function FmeanSquare_inplace(value,auxvalue,x...) # inplace
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(x[i].*x[i])
    end
    copy!(value,tmp/length(x))
end


DmeanSquare(derivativeIDX,f_c,faux_c,grad_c,grad_n,x...)=axpy!(2.0/(length(x)*length(x[derivativeIDX])),grad_c.*x[derivativeIDX],grad_n)


if PROC=="GPU"
    function FmeanSquare(x::CudaArray...)
        tmp=CudaArray(zeros(1))
        for i in 1:length(x)
            CUBLAS.gemv!('T',1.0/(length(x)*length(x[i])),flatten(Float64,x[i]),vec(x[i]),1.0,tmp)
        end
    return (tmp,nothing) # must always return an array for the value
    end

    function FmeanSquare_inplace(value,auxvalue,x::CudaArray...) # inplace
        fill!(value,0.0)
        for i in 1:length(x)
            CUBLAS.gemv!('T',1.0/(length(x)*length(x[i])),flatten(Float64,x[i]),vec(x[i]),1.0,value);
        end
    end

    function DmeanSquare(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray...)
        alphaaxpy!(2.0/(length(x)*length(x[derivativeIDX])),grad_c,x[derivativeIDX],grad_n)
    end

end

Inplace[FmeanSquare]=FmeanSquare_inplace
Derivative[FmeanSquare]=DmeanSquare

meanSquare(n::ADnode)=ADnode(FmeanSquare,n)

function meanSquare(n::ArrayADnode)
    return ADnode(FmeanSquare,n)
end

#meanSquare(A::ADtrans)=ADFunction(FmeanSquare, ftranspose(node[A.parent]))
meanSquare(A::ADtrans)=ADFunction(FmeanSquare, node[A.parent]) # meanSq(A')=meanSq(A)

export meanSquare
