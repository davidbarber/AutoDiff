# f(x)=elu(x)=x*[x>0] + (exp(x)-1)*[x<0]

alpha=1.0
function Felu(x)
    y=zeros(size(x))
    for i=1:length(x)
        if x[i]>0
            y[i]=x[i]
        else
            y[i]=alpha*(exp(x[i])-1.0)
        end
    end
    return (y,nothing)
end

function Felu_inplace(handle,value::Array,auxvalue,x) # inplace
    for i=1:length(x)
        if x[i]>0
            value[i]=x[i]
        else
            value[i]=alpha*(exp(x[i])-1.0)
        end
    end
end


function Delu(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x)
    tmp=ones(size(x))
    for i=1:length(x)
        if x[i]<0
            tmp[i]=f_c[i]+alpha
        end
    end
    axpy!(1.0,tmp.*grad_c,grad_n)    
end

if 1==0 # TODO GPU version

    function Felu_inplace(handle,value::CudaArray,auxvalue,x::CudaArray...) # inplace
    end

    function Delu(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray...)
    end

end

Derivative[Felu]=Delu # Define dictionary lookup
Inplace[Felu]=Felu_inplace

elu(n::ADnode)=ADFunction(Felu,n)

###elu(A::ADtrans)=transpose(elu(node[A.parent])) # elu(A')=(elu(A))' TODO:check

export elu
