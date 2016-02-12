# alex's funcion
c=0.5
c1mlogc=c*(1-log(c))

function alex(x::Array)
    ind1=x.>-c
    ind2=x.<=-c    
    tmp=zeros(size(x))
    tmp[ind1]=x[ind1]
    tmp[ind2]=-log(-x[ind2]).*c-c1mlogc
    ind1=ind2=0
    return tmp
end

function gradalex(x::Array)
    ind1=x.>-c
    ind2=x.<=-c    
    tmp=zeros(size(x))
    tmp[ind1]=1.0
    tmp[ind2]=-c./x[ind2]
    ind1=ind2=0
    return tmp
end


Falex(x::Array{Float64,2})=( alex(x) ,[]);

function Falex_inplace(value,auxvalue,x::Array{Float64,2})
    ind1=x.>-c
    ind2=x.<=-c    
    tmp=zeros(size(x))
    tmp[ind1]=x[ind1]+c1mlogc
    tmp[ind2]=-log(-x[ind2]).*c
    ind1=ind2=0
    copy!(value,tmp)
end

Dalex(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)=axpy!(1.0,grad_c.*gradalex(x),grad_n)

if PROC=="GPU" # TODO
    #Falex(x::CudaArray)=(alex(x),[])
    Falex_inplace(value,aux,x::CudaArray)=alex!(x,value)

    Dalex(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)=gradalex!(grad_c,x,grad_n)
end

Derivative[Falex]=Dalex
Inplace[Falex]=Falex_inplace

alex(A::ADnode)=ADnode(Falex,A)
export alex


