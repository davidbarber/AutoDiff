# kink linear: f(x)=max(x,gamma*x) where gamma=0.25
gamma=0.25
kinklin(x)=max(x,gamma*x)

Fkinklin(x::Array{Float64,2})=(max(x,gamma*x),nothing)
Fkinklin_inplace(value,auxvalue,x::Array{Float64,2})=copy!(value,max(x,gamma*x))

Dkinklin(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)=axpy!(1.0,grad_c.*(gamma+(1-gamma)*(x.>0)),grad_n)

if PROC=="GPU"
    Fkinklin_inplace(value,aux,x::CudaArray)=kinklin!(x,value)
    function Dkinklin(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)
        tmp=CudaArray(Float64,size(grad_c)); fill!(tmp,0.0)
        A_emult_Bg0!(grad_c,x,tmp)
        scale!((1-gamma),tmp)
        axpy!(gamma,grad_c,tmp)
        axpy!(1.0,tmp,grad_n)
        free(tmp)
    end
end

Derivative[Fkinklin]=Dkinklin
Inplace[Fkinklin]=Fkinklin_inplace

kinklin(A::ADnode)=ADnode(Fkinklin,A)
export kinklin
