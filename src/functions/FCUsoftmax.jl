#TODO CPU Version
function FCUsoftmax(inputs::Array)

println("CPU version of CuDNN softmax under Development")
return inputs,inputs
end
#TODO this is the CPU version
function DCUsoftmax()
println("CPU version of CuDNN softmax under Development")
end

function FCUsoftmax(handle,value::CudaArray,auxvalue,X::CudaArray)
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
cudnnSoftmaxForward(handle,1,1,alpha,srcDataDesc,X.ptr,beta,dstDataDesc,value.ptr)
return value
end




function DCUsoftmax(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray)
alpha = 1.0
beta = 0.0
(n,c,h,w) = size(X)
dtype = eltype(X)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)
temp = CudaArray(dtype,n,c,h,w)
diffDataDesc = cudnnCreateTensorDescriptor()
(n,c,h,w) = size(grad_c)
cudnnSetTensor4dDescriptor(diffDataDesc,dataType,n,c,h,w)

cudnnSoftmaxBackward(handle,1,1,alpha,srcDataDesc,X.ptr,diffDataDesc,grad_c.ptr,beta,srcDataDesc,temp)
CUBLAS.axpy!(1.0,temp,grad_n)
free(temp)
cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(diffDataDesc)
return grad_n
end


Derivative[FCUsoftmax] = DCUsoftmax
Inplace[FCUsoftmax]   = FCUsoftmax
CUsoftmax(i::ADnode)=ADFunction(FCUsoftmax,i)
export CUsoftmax
