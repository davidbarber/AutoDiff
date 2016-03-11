#TODO CPU version 
function FPooling(malloc::Bool,inputs)

# now assume 2*2 pooling and stride 2
(n,c,h,w) = size(inputs)

return (n,c,Int(ceil((h-2)/2)+1),Int(ceil((w-2)/2)+1))
end
#TODO this is the CPU version
function DPooling()


end


if PROC=="GPU"
function FPooling(handle,value::CudaArray,auxvalue,X::CudaArray)
free(value)
(n,c,h,w) = size(X)
dtype = eltype(X)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)
poolingDesc = cudnnCreatePoolingDescriptor()
cudnnSetPooling2dDescriptor(poolingDesc,0,2,2,0,0,2,2)
(n,c,h,w)=cudnnGetPooling2dForwardOutputDim(poolingDesc,srcDataDesc)
println("The estimate forward pooling size is")
println((n,c,h,w))
value = CudaArray(dtype,(n,c,h,w))
dstDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDataDesc,dataType,n,c,h,w)
alpha = 1.0
beta = 0.0
cudnnPoolingForward(handle,poolingDesc,alpha,srcDataDesc,X.ptr,beta,dstDataDesc,value.ptr)
cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(dstDataDesc)
cudnnDestroyPoolingDescriptor(poolingDesc)
return value
end

function DPooling(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,X::CudaArray)

poolingDesc = cudnnCreatePoolingDescriptor()
cudnnSetPooling2dDescriptor(poolingDesc,0,2,2,0,0,2,2)
alpha = 1.0
beta = 0.0
(n,c,h,w) = size(f_c)
println("dimension of srcDesc is")
println((n,c,h,w))
dtype = eltype(X)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)

diffDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(diffDataDesc,dataType,n,c,h,w)


(n,c,h,w) = size(X)
dstDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDataDesc,dataType,n,c,h,w)
temp = CudaArray(dtype,n,c,h,w)
println(size(temp))
println(size(grad_n))
println(size(grad_c))
cudnnPoolingBackward(handle,poolingDesc,alpha,srcDataDesc,f_c.ptr,diffDataDesc,grad_c.ptr,dstDataDesc,X.ptr,beta,dstDataDesc,temp.ptr)
CUBLAS.axpy!(1.0,temp,grad_n)
free(temp)
cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(diffDataDesc)
cudnnDestroyTensorDescriptor(dstDataDesc)
cudnnDestroyPoolingDescriptor(poolingDesc)
return grad_n

end

end
Derivative[FPooling] = DPooling
Inplace[FPooling]   = FPooling
Pooling(i::ADnode)=ADFunction(FPooling,i)

export Pooling