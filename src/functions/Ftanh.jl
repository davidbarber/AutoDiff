# f(x)=tanh(x)

sf=1.0

Ftanh(x::Array{Float64,2})=(sf*tanh(x),[]);
Ftanh_inplace(value,aux,x::Array{Float64,2})=copy!(value,sf*tanh(x))

Dtanh(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)=axpy!(sf,grad_c.*(1.-(f_c/sf).^2),grad_n)

if PROC=="GPU" ## TODO
    function Ftanh(x::CudaArray)
        tmp=copy(x)
        tanh!(sf,x,tmp)
        return (tmp,[]) # problem here is that I don't know how to free tmp -- this will cause memory leak I think.
    end
    Ftanh_inplace(value::CudaArray,aux,x::CudaArray)=tanh!(sf,x,value)

    Dtanh(derivativeIDX,f_c,faux_c,grad_c::CudaArray,grad_n::CudaArray,x::CudaArray)=Dtanh!(sf,grad_c,f_c,grad_n)
end

Derivative[Ftanh]=Dtanh
Inplace[Ftanh]=Ftanh_inplace

import Base.tanh
tanh(A::ADnode)=ADnode(Ftanh,A)
export tanh
