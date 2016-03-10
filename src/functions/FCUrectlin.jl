#TODO CPU Version

function FCrossChannel(malloc::Bool,inputs)

println("CPU rectLin in development")
return size(inputs)
end
#TODO this is the CPU version

function DCrossChannel()
println("CPU Version under development")
end



if PROC=="GPU"
function FCrossChannel(handle,value::CudaArray,auxvalue,X::CudaArray)
#creation 

free(value)
(n,c,h,w) = size(X)
dtype = eltype(X)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)
value = CudaArray(dtype,(n,c,h,w))


normDesc = cudnnCreateLRNDescriptor()
N = 5
alpha = 0.0001
beta = 0.75
K = 1.0
cudnnSetLRNDescriptor(normDesc,N,alpha,beta,K)
dstDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDataDesc,dataType,n,c,h,w)
cudnnLRNCrossChanelForward(handle,normDesc,0,alpha,srcDataDesc,X.ptr,beta,dstDataDesc,value.ptr)

cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(dstDataDesc)
cudnnDestroyLRNDescriptor(normDesc)
return value
end

function DCrossChannel(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,X::CudaArray)
# grad_n child
# grad_c current 
normDesc = cudnnCreateLRNDescriptor()
N = 5
alpha = 0.0001
beta = 0.75
K = 1.0
cudnnSetLRNDescriptor(normDesc,N,alpha,beta,K)

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

cudnnLRNCrossChannelBackward(handle,normDesc,0,alpha,srcDataDesc,f_c.ptr,diffDataDesc,grad_c.ptr,dstDataDesc,X.ptr,beta,dstDataDesc,temp.ptr)
CUBLAS.axpy!(1.0,temp,grad_n)

free(temp)
cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(diffDataDesc)
cudnnDestroyTensorDescriptor(dstDataDesc)
cudnnDestroyLRNDescriptor(normDesc)
return grad_n
end

end

Derivative[FCrossChannel] = DCrossChannel
Inplace[FCrossChannel]   = FCrossChannel
CrossChannel(i::ADnode)=ADFunction(FCrossChannel,i)
export CrossChannel


