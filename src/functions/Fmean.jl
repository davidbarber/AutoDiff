# f(x)=mean(x)

function Fmean(x...)
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(x[i])
    end
    return ([tmp/length(x)],nothing)
end

function Fmean_inplace(handle,value::Array,auxvalue,x...) # inplace
    tmp=0.0
    for i in 1:length(x)
        tmp+=mean(x[i])
    end
    fill!(value,tmp/length(x))
end


function Dmean(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x...)
    axpy!(grad_c[1],ones(size(grad_n))/(length(x)*length(x[derivativeIDX])),grad_n)
end


import Base.mean

if PROC=="GPU"

    function mean(A::CudaArray{Float64})
        return flatten(Float64,CUBLAS.gemv('T',1./length(A),flatten(Float64,A),CudaArray(ones(length(A)))))
    end

    function mean(A::CudaArray{Float32})
        return flatten(Float32,CUBLAS.gemv('T',Float32(1./length(A)),flatten(Float32,A),CudaArray(Float32,ones(Float32,length(A)))))
    end    
    export mean

    function mean!(A::CudaArray{Float64},Out::CudaArray{Float64})
         CUBLAS.gemv!('T',1./length(A),flatten(Float64,A),CudaArray(ones(length(A))),0.0,Out)
    end

    function mean!(A::CudaArray{Float32},Out::CudaArray{Float32})
         CUBLAS.gemv!('T',Float32(1./length(A)),flatten(Float32,A),CudaArray(Float32,ones(length(A))),Float32(0.0),Out)
    end
    
    export mean!


    function Fmean()
    end
    
    function Fmean(x::CudaArray{Float64}...)
        tmp=CudaArray(zeros(1,1))
        for i in 1:length(x)
            axpy!(1.0/length(x),mean(x[i]),tmp)
        end
        return (tmp,nothing)
    end

    function Fmean(x::CudaArray{Float32}...)
        tmp=CudaArray(Float32,zeros(1,1))
        for i in 1:length(x)
            axpy!(Float32(1.0/length(x)),mean(x[i]),tmp)
        end
        return (tmp,nothing)
    end
    
    function Fmean_inplace(handle,value::CudaArray{Float64},auxvalue,x::CudaArray{Float64}...) # inplace

        fill!(value,0.0)
        for i in 1:length(x)
            axpy!(1.0/length(x),mean(x[i]),value)
        end
    end

    function Fmean_inplace(handle,value::CudaArray{Float32},auxvalue,x::CudaArray{Float32}...) # inplace
        fill!(value,Float32(0.0))
        for i in 1:length(x)
            axpy!(Float32(1.0/length(x)),mean(x[i]),value)
        end
    end


    function Dmean(handle,derivativeIDX,f_c,faux_c,grad_c::CudaArray{Float64},grad_n::CudaArray{Float64},x::CudaArray{Float64}...)

        tmp=CudaArray(Float64,size(grad_n))
        fill!(tmp,1.0/(length(x)*length(x[derivativeIDX])))
        axpy!(grad_c,tmp,grad_n)
        free(tmp)
    end

    function Dmean(handle,derivativeIDX,f_c,faux_c,grad_c::CudaArray{Float32},grad_n::CudaArray{Float32},x::CudaArray{Float32}...)
        tmp=CudaArray(Float32,size(grad_n))
        fill!(tmp,Float32(1.0/(length(x)*length(x[derivativeIDX]))))
        axpy!(grad_c,tmp,grad_n)
        free(tmp)
    end

end

Derivative[Fmean]=Dmean # Define dictionary lookup
Inplace[Fmean]=Fmean_inplace

mean(n::ADnode)=ADFunction(Fmean,n)

function mean(n::ArrayADnode)
    return ADFunction(Fmean,n...)
end


#mean(A::ADtrans)=ADFunction(Fmean, ftranspose(node[A.parent]))
mean(A::ADtrans)=ADFunction(Fmean, node[A.parent]) # mean(A')=mean(A)

export mean
