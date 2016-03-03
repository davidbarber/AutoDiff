# f(x)=abs(x)




function Fabs(x)
return size(x)
end

function Fabs(x)
    return (abs(x),nothing)
end

function Fabs_inplace(handle,value::Array,auxvalue,x) # inplace
    copy!(value,abs(x))
end


function Dabs(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x)
    axpy!(1.0,sign(x).*grad_c,grad_n)
end

if PROC=="GPU" 
    function Fabs_inplace(handle,value::CudaArray,auxvalue,x::CudaArray) # inplace
        fill!(value,0.0)
        abs!(x,value)
    end

    function Dabs(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)
        xsigny_update!(grad_c,x,grad_n)
    end
end

Derivative[Fabs]=Dabs # Define dictionary lookup
Inplace[Fabs]=Fabs_inplace

import Base.abs

abs(n::ADnode)=ADFunction(Fabs,n)

###abs(A::ADtrans)=transpose(abs(node[A.parent])) # abs(A')=(abs(A))' TODO:check

export abs
