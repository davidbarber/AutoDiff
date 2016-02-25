# BinaryEntropy loss: f(x,y)=sum(x.*log(x./y)+(1.-x).*log((1.-x)./(1.-y)))/prod(size(x))
# It's often a bad idea to use this function since it is numerically unstable.
# BinaryKullbackLeiblerLossXsigmaY is usually better.

#FBinaryEntropyLoss(x::Array,y::Array)=([sum(x.*log(x./y)+(1.-x).*log((1.-x)./(1.-y)))/length(x)],[])

FBinaryEntropyLoss(x::Array,y::Array)=([sum(x.*log(x./y)+(1.-x).*log((1.-x)./(1.-y)))/length(x)],((y-x)./(y.*(1.-y))/length(x)))

#FBinaryEntropyLoss_inplace(value,aux,x::Array,y::Array)=copy!(value,sum(x.*log(x./y)+(1.-x).*log((1.-x)./(1.-y)))/length(x))
function FBinaryEntropyLoss_inplace(value,aux,x::Array,y::Array)
    copy!(value,sum(x.*log(x./y)+(1.-x).*log((1.-x)./(1.-y)))/length(x))
    copy!(aux,((y-x)./(y.*(1.-y)))/length(x))
end

function DBinaryEntropyLoss(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x,y)
    if derivativeIDX==1
        axpy!(grad_c[1],log(x.*(1.-y)./(y.*(1.-x)))/length(x),grad_n)
    elseif derivativeIDX==2
        #axpy!(grad_c[1],((y-x)./(y.*(1.-y)))/length(x),grad_n)
        axpy!(grad_c[1],faux_c,grad_n)
    end
end


if PROC=="GPU"
    FBinaryEntropyLoss(x::CudaArray,y::CudaArray)=(binaryentropy(x,y),nothing)
    FBinaryEntropyLoss_inplace(handle,value,aux,x::CudaArray,y::CudaArray)=copy!(value,binaryentropy(x,y))

    function DBinaryEntropyLoss(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        if derivativeIDX==1
            DXbinaryentropy!(x,y,grad_c,grad_n)
        elseif derivativeIDX==2
            DYbinaryentropy!(x,y,grad_c,grad_n)
        end
    end
end

Derivative[FBinaryEntropyLoss]=DBinaryEntropyLoss
Inplace[FBinaryEntropyLoss]=FBinaryEntropyLoss_inplace

BinaryKullbackLeiblerLoss(nx,ny)=ADFunction(FBinaryEntropyLoss,nx,ny)
export BinaryKullbackLeiblerLoss

