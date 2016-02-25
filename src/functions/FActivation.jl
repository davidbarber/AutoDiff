#TODO CPU Version
function FActivation(inputs::Array)

println("CPU version under development")
return inputs,inputs
end
#TODO this is the CPU version
function DActivation()


end


#TODO missing handler
function FActivation(value::CudaArray,au::CudaArray,tensor::CudaTensor)
dstTensorDesc = cudnnCreateTensorDescriptor()
dimension = tensor.dims
cudnnSetTensor4dDescriptor(dstTensorDesc,tensor.dataType,dimension[1],dimension[2],dimension[3],dimension[4])
alpha = 1.0
beta = 0.0
cudnnActivationForward(tensor.handle,1,alpha,tensor.tensorDesc,tensor.data.ptr,beta,dstTensorDesc,value.ptr)
end

function DActivation()


end

Derivative[FActivation] = DActivation
Inplace[FActivation]   = FActivation
CUActivation(i::ADTensor)=ADFunction(FActivation,i)
export CUActivation


#TODO: still need to be change to fit the ADnode 
function activationBackward(inputs::CudaArray,outputs::CudaArray,diffDesc::cudnnTensorDescriptor_t,diff::CudaArray,n,c,h,w)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,n*c*w*h)
cudnnActivationBackward(ctx.handle,1,alpha,ctx.dstTensorDesc,ouputs,diffDesc,diff,ctx.srcTensorDesc,inputs,beta,diffDesc,diff,ctx.srcTensorDesc,out)
return out
end