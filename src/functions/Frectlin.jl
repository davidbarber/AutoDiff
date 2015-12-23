# rectified linear: f(x)=max(x,0)
rectlin(x)=x.*(x.>0)

Frectlin(x::Array{Float64,2})=(max(x,0),[]);
Frectlin_inplace(value,auxvalue,x::Array{Float64,2})=copy!(value,max(x,0))

Drectlin(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)=axpy!(1.0,grad_c.*(x.>0),grad_n)

if PROC=="GPU" # TODO
    Frectlin(x::CudaArray)=(rectlin(x),[])
    Frectlin_inplace(value,aux,x::CudaArray)=rectlin!(x,value)

    Drectlin(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)=A_emult_Bg0!(grad_c,x,grad_n)#; t.*(x.>0)
end

Derivative[Frectlin]=Drectlin
Inplace[Frectlin]=Frectlin_inplace

rectlin(A::ADnode)=ADnode(Frectlin,A)
export rectlin


