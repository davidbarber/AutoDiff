#f(x,y)=mean(KL(x,sigmoid(y))), where KL is the Kullback-Leibler divergence

FBinaryEntropyLossXsigmoidY(x,y)=([mean(x.*log(x)+(1.-x).*log(1.-x)+x.*y-log1pexp(y))],nothing)
FBinaryEntropyLossXsigmoidY_inplace(value,aux,x,y)=copy!(value,mean(x.*log(x)+(1.-x).*log(1.-x)+x.*y-log1pexp(y)))

DBinaryEntropyLossXsigmoidY[1]

function DBinaryEntropyLossXsigmoidY(derivativeIDX,f_c,faux_c,grad_c,grad_n,x,y)
    if derivativeIDX==1
        axpy!(grad_c[1]/length(x),y,grad_n)
    elseif derivativeIDX==2
        axpy!(grad_c[1]/length(x),x-sigmoid(y)),grad_n)
    end
end

Derivative[FBinaryEntropyLossXsigmoidY]=DBinaryEntropyLossXsigmoidY
Inplace[FBinaryEntropyLossXsigmoidY]=DBinaryEntropyLossXsigmoidY_inplace

BinaryKullbackLeiblerLossXsigmoidY(nx,ny)=ADnode(FbinaryEntropyLoss,[nx ny])
export BinaryKullbackLeiblerLossXsigmoidY

#TODO: GPU
