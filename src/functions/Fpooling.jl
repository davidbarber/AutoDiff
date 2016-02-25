#TODO CPU version 
function FPooling(inputs::Array)

println("CPU version under develop")
return inputs,inputs
end
#TODO this is the CPU version
function DPooling()


end


function FPooling(value::CudaArray,au::CudaArray,tensor::CudaTensor)
const dims = 2
windowDim = [2,2]
padding = [0,0]
stride = [2,2]
poolingDesc = cudnnCreatePoolingDescriptor()
cudnnSetPooling2dDescriptor(poolingDesc,0,2,2,0,0,2,2)
(n,c,h,w)=cudnnGetPooling2dForwardOutputDim(poolingDesc,tensor.tensorDesc)
dstDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDesc,tensor.dataType,n,c,h,w)
alpha = 1.0
beta = 0.0
cudnnPoolingForward(tensor.handle,poolingDesc,alpha,tensor.tensorDesc,tensor.data.ptr,beta,dstDesc,value.ptr)

end

function DPooling()


end

Derivative[FPooling] = DPooling
Inplace[FPooling]   = FPooling
Pooling(i::ADTensor)=ADFunction(FPooling,i)
export Pooling