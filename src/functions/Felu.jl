# f(x)=elu(x)

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

function Felu_inplace(value::Array,auxvalue,x) # inplace
    for i=1:length(x)
        if x[i]>0
            value[i]=x[i]
        else
            value[i]=alpha*(exp(x[i])-1.0)
        end
    end
end


function Delu(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)
    tmp=ones(size(x))
    for i=1:length(x)
        if x[i]<0
            tmp[i]=f_c[i]+alpha
        end
    end
    axpy!(1.0,tmp.*grad_c,grad_n)    
end

if 1==0 # TODO

    function uuelu(A::CudaArray)
        return CudaArray(CUBLAS.asum(A)/length(A)*ones(1,1))
    end
    export meanElu

    function meanElu!(A::CudaArray,Out::CudaArray)
        copy!(Out,meanElu(A))
    end
    export meanElu!


    function FmeanElu(x::CudaArray...)
        tmp=CudaArray(zeros(1,1))
        for i in 1:length(x)
            axpy!(1.0/length(x),meanElu(x[i]),tmp)
        end
        return (tmp,nothing)
    end


    function FmeanElu_inplace(value::CudaArray,auxvalue,x::CudaArray...) # inplace
        fill!(value,0.0)
        for i in 1:length(x)
            axpy!(1.0/length(x),meanElu(x[i]),value)
        end
    end


    function DmeanElu(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray...)
        tmp=CudaArray(Float64,size(grad_n))
        vsign!(x[derivativeIDX],tmp)
        alphaaxpy!(1.0/(length(x)*length(x[derivativeIDX])),grad_c,tmp,grad_n)
        free(tmp)
    end

end

Derivative[Felu]=Delu # Define dictionary lookup
Inplace[Felu]=Felu_inplace

elu(n::ADnode)=ADnode(Felu,n)

###elu(A::ADtrans)=transpose(elu(node[A.parent])) # elu(A')=(elu(A))' TODO:check

export elu
