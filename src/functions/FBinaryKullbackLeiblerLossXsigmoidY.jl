#f(x,y)=mean(KL(x,sigmoid(y))), where KL is the Kullback-Leibler divergence

FBinaryEntropyLossXsigmoidY(x,y)=([sum(x.*log(x)+(1.-x).*log(1.-x)-x.*y+log1pexp(y))/length(x)],nothing)
FBinaryEntropyLossXsigmoidY_inplace(value,aux,x,y)=copy!(value,sum(x.*log(x)+(1.-x).*log(1.-x)-x.*y+log1pexp(y))/length(x))

function DBinaryEntropyLossXsigmoidY(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x,y)
    if derivativeIDX==1
        axpy!(-grad_c[1]/length(x),y-log(x./(1-x)),grad_n)
    elseif derivativeIDX==2
        axpy!(grad_c[1]/length(x),sigmoid(y)-x,grad_n)
    end
end


if PROC=="GPU"
#    FBinaryEntropyLossXsigmoidY(x::CudaArray,y::CudaArray)=(binaryentropyXsigmoidY(x,y),nothing)
    FBinaryEntropyLossXsigmoidY_inplace(value,aux,x::CudaArray,y::CudaArray)=binaryentropyXsigmoidY!(x,y,value)
    
    function DBinaryEntropyLossXsigmoidY(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        if derivativeIDX==1
            DXbinaryentropyXsigmoidY!(x,y,grad_c,grad_n)
        elseif derivativeIDX==2
            DYbinaryentropyXsigmoidY!(x,y,grad_c,grad_n)
        end
    end

#    FBinaryEntropyLossXsigmoidY_inplace(value,aux,x::CudaArray,y::CudaArray)=binaryentropyXsigmoidY!(x,y,value)
    
#    function DBinaryEntropyLossXsigmoidY(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray{Float64},y::CudaArray{Float64})
#        if derivativeIDX==1
#            DXbinaryentropyXsigmoidY!(x,y,grad_c,grad_n)
#        elseif derivativeIDX==2
#            DYbinaryentropyXsigmoidY!(x,y,grad_c,grad_n)
#        end
#    end
   

#    function DBinaryEntropyLossXsigmoidY(derivativeIDX,f_c,faux_c,grad_c::CudaArray{Float32},grad_n::CudaArray{Float32},x::CudaArray{Float32},y::CudaArray{Float32})
#        if derivativeIDX==1
#            DXbinaryentropyXsigmoidY_32!(x,y,grad_c,grad_n)
#        elseif derivativeIDX==2
#            DYbinaryentropyXsigmoidY_32!(x,y,grad_c,grad_n)
#        end
#    end    

    
end




Derivative[FBinaryEntropyLossXsigmoidY]=DBinaryEntropyLossXsigmoidY
Inplace[FBinaryEntropyLossXsigmoidY]=FBinaryEntropyLossXsigmoidY_inplace

BinaryKullbackLeiblerLossXsigmoidY(nx,ny)=ADFunction(FBinaryEntropyLossXsigmoidY,nx,ny) # give it a better name
export BinaryKullbackLeiblerLossXsigmoidY


