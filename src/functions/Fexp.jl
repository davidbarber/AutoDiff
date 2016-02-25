Fexp(x)=(exp(x),nothing)

function Fexp_inplace(value,auxvalue,x)
    copy!(value,exp(x))
end

function Dexp(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)
    axpy!(1.0,grad_c.*f_c,grad_n)
end


if PROC=="GPU"

    function Fexp(x::CudaArray)
        tmp=CudaArray(Float64,size(x))
        exp!(x,tmp)
        return (tmp,nothing) # memory leak here
    end
    
    function Fexp_inplace(value,auxvalue,x::CudaArray)
        exp!(x,value)
    end
    
    function Dexp(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)
        vmultupdate!(1.0,grad_c,f_c,grad_n)
    end

end



Derivative[Fexp]=Dexp
Inplace[Fexp]=Fexp_inplace

import Base.exp

exp(n::ADnode)=ADFunction(Fexp,n)

export exp
