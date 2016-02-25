#TODO CPU Version
function FActivation(inputs::Array)

println("CPU version under development")
return inputs,inputs
end
#TODO this is the CPU version
function DActivation()


end


if PROC=="GPU"
function FActivation(handle,value::CudaArray,auxvalue,X::CudaArray)

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


cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(dstDataDesc)
return value
end

function DActivation(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,X::CudaArray)
alpha = 1.0
beta = 0.0
(n,c,h,w) = size(f_c)
dtype = eltype(f_c)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)

(n,c,h,w) = size(X)
dstDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDataDesc,dataType,n,c,h,w)
temp = CudaArray(dtype,n,c,h,w)

(n,c,h,w) = size(grad_c)
diffDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(diffDataDesc,dataType,n,c,h,w)

cudnnActivationBackward(handle,1,alpha,srcDataDesc,f_c.ptr,diffDataDesc,grad_c.ptr,dstDataDesc,X.ptr,beta,dstDataDesc,temp.ptr)
CUBLAS.axpy!(1.0,temp,grad_n)

cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(diffDataDesc)
cudnnDestroyTensorDescriptor(dstDataDesc)
return grad_n
end

end
Derivative[FActivation] = DActivation
Inplace[FActivation]   = FActivation
CUActivation(i::ADnode)=ADFunction(FActivation,i)
export CUActivation

