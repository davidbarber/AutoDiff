#f(x,y)=mean(KL(x,sigmoid(y))), where KL is the Kullback-Leibler divergence

FBinaryEntropyLossXsigmoidY(x,y)=([sum(x.*log(x)+(1.-x).*log(1.-x)-x.*y+log1pexp(y))/length(x)],nothing)
FBinaryEntropyLossXsigmoidY_inplace(value,aux,x,y)=copy!(value,sum(x.*log(x)+(1.-x).*log(1.-x)-x.*y+log1pexp(y))/length(x))

function DBinaryEntropyLossXsigmoidY(derivativeIDX,f_c,faux_c,grad_c,grad_n,x,y)
    if derivativeIDX==1
        axpy!(-grad_c[1]/length(x),y-log(x./(1-x)),grad_n)
    elseif derivativeIDX==2
        axpy!(grad_c[1]/length(x),sigmoid(y)-x,grad_n)
    end
end


if PROC=="GPU"
#    FBinaryEntropyLossXsigmoidY(x::CudaArray,y::CudaArray)=(binaryentropyXsigmoidY(x,y),nothing)
    FBinaryEntropyLossXsigmoidY_inplace(value,aux,x::CudaArray,y::CudaArray)=binaryentropyXsigmoidY!(x,y,value)
    
    function DBinaryEntropyLossXsigmoidY(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        if derivativeIDX==1
            DXbinaryentropyXsigmoidY!(x,y,grad_c,grad_n)
        elseif derivativeIDX==2
            DYbinaryentropyXsigmoidY!(x,y,grad_c,grad_n)
        end
    end    
end


Derivative[FBinaryEntropyLossXsigmoidY]=DBinaryEntropyLossXsigmoidY
Inplace[FBinaryEntropyLossXsigmoidY]=FBinaryEntropyLossXsigmoidY_inplace

BinaryKullbackLeiblerLossXsigmoidY(nx,ny)=ADnode(FBinaryEntropyLossXsigmoidY,[nx ny]) # give it a better name
export BinaryKullbackLeiblerLossXsigmoidY


