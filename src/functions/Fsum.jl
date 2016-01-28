# Sum function: f(x)=sum(x)
Fsum(x)=([sum(x)],nothing)
Fsum_inplace(handle,value,auxvalue,x)=fill!(value,sum(x))

function Dsum(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x_n...)
    axpy!(grad_c[1],ones(size(grad_n)),grad_n)
end

function Fsum(x...)
    tmp=0.0
    for i in 1:length(x)
        tmp+=sum(x[i])
    end
    return ([tmp],nothing)
end

function Fsum_inplace(handle,value::Array,auxvalue,x...) # inplace
    tmp=0.0
    for i in 1:length(x)
        tmp+=sum(x[i])
    end
    fill!(value,tmp)
end


import Base.sum


if PROC=="GPU"

    function sum(A::CudaArray)
        return flatten(Float64,CUBLAS.gemv('T',1.,flatten(Float64,A),CudaArray(ones(length(A)))))
    end
    export sum

    import Base.sum!
    function sum!(out::CudaArray,A::CudaArray)
        CUBLAS.gemm!('T','N',1.,flatten(Float64,A),CudaArray(ones(length(A),1)),0.0,out)
    end
    export sum!


    function sum_update!(alpha::Float64,A::CudaArray,beta::Float64,out::CudaArray)
        CUBLAS.gemm!('T','N',alpha,flatten(Float64,A),CudaArray(ones(length(A),1)),beta,out)
    end
    export sum_update!


    function Dsum(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray...)
        tmp=CudaArray(Float64,size(grad_n))
        fill!(tmp,1.0)
        axpy!(grad_c,tmp,grad_n)
        free(tmp)
    end


    function Fsum(x::CudaArray...)
        tmp=CudaArray(zeros(1,1))
        for i in 1:length(x)
            axpy!(1.0,sum(x[i]),tmp)
        end
        return (tmp,nothing)
    end


    function Fsum_inplace(handle,value::CudaArray,auxvalue,x::CudaArray...) # inplace
        fill!(value,0.0)
        for i in 1:length(x)
            axpy!(1.0,sum(x[i]),value)
        end
    end


end

Derivative[Fsum]=Dsum # Define dictionary lookup
Inplace[Fsum]=Fsum_inplace

import Base.sum
sum(n::ADnode)=ADFunction(Fsum,n)


function sum(n::ArrayADnode)
    return ADnode(Fsum,n)
end

# TODO: add similar as below for each unary function that can take a transposed argument
#sum(A::ADtrans)=ADFunction(Fsum, ftranspose(node[A.parent]))
sum(A::ADtrans)=ADFunction(Fsum, node[A.parent]) # sum(A')=sum(A)

sum(A::ADdiag)=sum(node[A.parent])

export sum
