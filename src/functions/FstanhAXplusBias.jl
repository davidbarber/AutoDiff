# Standard stanh layer: f(A,x,b)=stanh(A*x+b)
# Note that the bias (a column vector) gets expanded here to match the dimension of A*x
sf=2.5

FstanhAXplusBias(A,X,b)=begin; a=A*X+b*ones(1,size(X,2)); return (sf*tanh(a),[]); end

function FstanhAXplusBias_inplace(value,aux,A,X,b)
    a=A*X+b*ones(1,size(X,2))
    copy!(value,sf*tanh(a))
end

function DstanhAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X,b)
    
    if derivativeIDX==1
        axpy!(1.0,(sf*grad_c.*(1.0-(f_c/sf).^2))*X',grad_n)
    elseif derivativeIDX==2
        axpy!(1.0,sf*A'*(grad_c.*(1.0-(f_c/sf).^2)),grad_n)
    elseif derivativeIDX==3
        axpy!(1.0,sf*sum(grad_c.*(1.0-(f_c/sf).^2),2),grad_n)
    end
end

if PROC=="GPU"    
    function FstanhAXplusBias(A::CudaArray,X::CudaArray,b::CudaArray)       
        tmp=FAXplusBias(A,X,b)[1]
        stanh!(sf,tmp,tmp)
        return(tmp,[])
    end
    
    function FstanhAXplusBias_inplace(value,aux,A::CudaArray,X::CudaArray,b::CudaArray)
        stanh!(sf,FAXplusBias(A,X,b)[1],value)
    end
    
    function DstanhAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray,b::CudaArray)
        tmp=CudaArray(Float64,size(grad_c)); fill!(tmp,0.0)
        Dstanh!(sf,grad_c,f_c,tmp)
        if derivativeIDX==1
            gemm!('N','T',1.0,tmp,X,1.0,grad_n)
        elseif derivativeIDX==2
            gemm!('T','N',1.0,A,tmp,1.0,grad_n)
        elseif derivativeIDX==3
            ons=CudaArray(Float64,(size(X,2),1)); fill!(ons,1.0)
            gemm!('N','N',1.0,tmp,ons,1.0,grad_n)
            free(ons)
        end
        free(tmp)
    end
end

Derivative[FstanhAXplusBias]=DstanhAXplusBias
Inplace[FstanhAXplusBias]=FstanhAXplusBias_inplace

stanhAXplusBias(A,X,b)=ADFunction(FstanhAXplusBias,A,X,b)
export stanhAXplusBias


