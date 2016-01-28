#TODO CPU Version
function FCUsoftmax!(inputs::Array)

println("CPU version of CuDNN softmax under Development")
return inputs,inputs
end
#TODO this is the CPU version
function DCUsoftmax()
println("CPU version of CuDNN softmax under Development")
end

function FCUsoftmax!(handle,value::CudaArray,auxvalue,X::CudaArray)
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
cudnnSoftmaxBackward(tensor.handle,1,1,alpha,tensor.tensorDesc,tensor.data.ptr,tensor.tensorDesc,grad_c.ptr,beta,tensor.tensorDesc,grad_n.ptr)
end


Derivative[FCUsoftmax!] = DCUsoftmax
Inplace[FCUsoftmax!]   = FCUsoftmax!
CUsoftmax(i::ADnode)=ADFunction(FCUsoftmax!,i)
export CUsoftmax
