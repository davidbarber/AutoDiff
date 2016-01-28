#TODO CPU version 
function FPooling!(inputs::Array)

println("CPU version under develop")
return inputs,inputs
end
#TODO this is the CPU version
function DPooling()


end


function FPooling!(handle,value::CudaArray,auxvalue,X::CudaArray)
free(value)
(n,c,h,w) = size(X)
dtype = eltype(X)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)
poolingDesc = cudnnCreatePoolingDescriptor()
cudnnSetPooling2dDescriptor(poolingDesc,0,2,2,0,0,2,2)
(n,c,h,w)=cudnnGetPooling2dForwardOutputDim(poolingDesc,srcDataDesc)
value = CudaArray(dtype,(n,c,h,w))
dstDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDataDesc,dataType,n,c,h,w)
alpha = 1.0
beta = 0.0
cudnnPoolingForward(handle,poolingDesc,alpha,srcDataDesc,X.ptr,beta,dstDataDesc,value.ptr)
return value
end

function DPooling(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,X::CudaArray)


end

Derivative[FPooling!] = DPooling
Inplace[FPooling!]   = FPooling!
Pooling(i::ADnode)=ADFunction(FPooling!,i)
export Pooling