#f(p,x)=KL(p,softmax(x))
function FKLsoftmax(malloc::Bool,p::Array{Float64,2},x::Array{Float64,2})
return (1,1)
end

FKLsoftmax(p::Array{Float64,2},x::Array{Float64,2})=begin DN=prod(size(x));Z=sum(exp(x),1);aux=(Z,DN); return ((sum(p.*(log(p)-x))+sum(log(Z)))/DN,aux); end

function FKLsoftmax_inplace(handle,value,auxvalue,p::Array{Float64,2},x::Array{Float64,2})
    DN=prod(size(x));Z=sum(exp(x),1);aux=(Z,DN) 
    value=((sum(p.*(log(p)-x))+sum(log(Z)))/DN,aux)
end


function DKLsoftmax(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::Array{Float64,2})
    if derivativeIDX==1
        grad_c.*(1+log(p)-x) 
    elseif derivativeIDX==2
        grad_c.*(exp(x)./aux[1]-p)./aux[2]
    end
end

if PROC=="GPU"
end


Derivative[FKLsoftmax]=DKLsoftmax
Inplace[FKLsoftmax]=FKLsoftmax_inplace

KLsoftmax(p::ADnode,x::ADnode)=ADFunction(FKLsoftmax,p,x)

export KLsoftmax
