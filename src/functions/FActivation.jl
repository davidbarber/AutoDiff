#TODO CPU Version
function FActivation!(inputs::Array)

println("CPU version under development")
return inputs,inputs
end
#TODO this is the CPU version
function DActivation()


end


#TODO missing handler
function FActivation!(handle,value::CudaArray,auxvalue,X::CudaArray)

free(value)
(n,c,h,w) = size(X)
dtype = eltype(X)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)
value = CudaArray(dtype,(n,c,h,w))
dstDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDataDesc,dataType,n,c,h,w)
alpha = 1.0
beta = 0.0
cudnnActivationForward(handle,1,alpha,srcDataDesc,X.ptr,beta,dstDataDesc,value.ptr)
return value
end

function DActivation(handle,)


end

Derivative[FActivation!] = DActivation
Inplace[FActivation!]   = FActivation!
CUActivation(i::ADnode)=ADFunction(FActivation!,i)
export CUActivation

