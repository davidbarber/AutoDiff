# f(x)=1./(1+exp(-x))
Fsigmoid(x)=(sigmoid(x),[]);
Fsigmoid_inplace(value,aux,x)=copy!(value,sigmoid(x))

Dsigmoid(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)=axpy!(1.0,grad_c.*f_c.*(1.-f_c),grad_n)

if PROC=="GPU" 
    function Fsigmoid(x::CudaArray)
        tmp=copy(x)
        sigmoid!(x,tmp)
        return (tmp,nothing) ## pproblem here is that I don't know how to free tmp -- this will cause memory leak I think.
    end
    Fsigmoid_inplace(value,aux,x::CudaArray)=sigmoid!(x,value)
    
    Dsigmoid(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)=tx1mx!(grad_c,f_c,grad_n)

end

Derivative[Fsigmoid]=Dsigmoid
Inplace[Fsigmoid]=Fsigmoid_inplace

sigmoid(A::ADnode)=ADFunction(Fsigmoid,A)
export sigmoid
