# f(x)=sf*tanh(x)

sf=2.5

Fstanh(x::Array{Float64,2})=(sf*tanh(x),[]);
Fstanh_inplace(value,aux,x::Array{Float64,2})=copy!(value,sf*tanh(x))

Dstanh(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)=axpy!(sf,grad_c.*(1.-(f_c/sf).^2),grad_n)

if PROC=="GPU" ## TODO
    function Fstanh(x::CudaArray)
        tmp=copy(x)
        stanh!(sf,x,tmp)
        return (tmp,[]) # problem here is that I don't know how to free tmp -- this will cause memory leak I think.
    end
    Fstanh_inplace(value::CudaArray,aux,x::CudaArray)=stanh!(sf,x,value)
    
    Dstanh(derivativeIDX,f_c,faux_c,grad_c::CudaArray,grad_n::CudaArray,x::CudaArray)=Dstanh!(sf,grad_c,f_c,grad_n)
end

Derivative[Fstanh]=Dstanh
Inplace[Fstanh]=Fstanh_inplace

stanh(A::ADnode)=ADnode(Fstanh,A)
export stanh
