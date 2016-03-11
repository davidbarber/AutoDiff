# rectified linear: f(x)=max(x,0)
rectlin(x)=x.*(x.>0)
function Frectlin(malloc::Bool,x::Array{Float64,2})
return size(x)
end

Frectlin(x::Array{Float64,2})=(max(x,0),[]);
Frectlin_inplace(handle,value,auxvalue,x::Array{Float64,2})=copy!(value,max(x,0))

Drectlin(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x)=axpy!(1.0,grad_c.*(x.>0),grad_n)

if PROC=="GPU" 
    Frectlin(x::CudaArray)=(rectlin(x),[])
    Frectlin_inplace(handle,value,aux,x::CudaArray)=rectlin!(x,value)

    Drectlin(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)=A_emult_Bg0!(grad_c,x,grad_n)#; t.*(x.>0)
end

Derivative[Frectlin]=Drectlin
Inplace[Frectlin]=Frectlin_inplace

rectlin(A::ADnode)=ADFunction(Frectlin,A)
export rectlin


