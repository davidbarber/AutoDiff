#TODO this is the CPU version 
function FConvolution(inputs::Array,filters::Array)
println("CPU version under develop")
return inputs, filters
end
#TODO this is the CPU version
function DConvolution()
println("CPU version under develop")
return 0
end


function FConvolution(value::CudaArray,au::CudaArray,tensor::CudaTensor,filter::CudaFilter)
convDesc = cudnnCreateConvolutionDescriptor()
cudnnSetConvolution2dDescriptor(convDesc,0,0,1,1,1,1,0)
(n,c,h,w)= cudnnGetConvolution2dForwardOutputDim(convDesc,tensor.tensorDesc,filter.filterDesc)
dstTensorDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstTensorDesc,tensor.dataType,n,c,h,w)
algo = cudnnGetConvolutionForwardAlgorithm(tensor.handle,tensor.tensorDesc,filter.filterDesc,convDesc,dstTensorDesc,1,0)
biasDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(biasDesc,tensor.dataType,1,c,1,1)


alpha =1.0
beta = 0.0
sizeInByte = cudnnGetConvolutionForwardWorkspaceSize(tensor.handle,tensor.tensorDesc,filter.filterDesc,convDesc,dstTensorDesc,algo)
workspace = nothing
if(sizeInByte ==0)
workspace = CudaPtr()
else
workspace =CUDArt.malloc(sizeInByte)
end
cudnnConvolutionForward(tensor.handle,alpha,tensor.tensorDesc,tensor.data.ptr,filter.filterDesc,filter.data.ptr,convDesc,algo,workspace,sizeInByte,beta,dstTensorDesc,value.ptr)

end

function DConvolution(tensor::CudaTensor,filter::CudaFilter)
println("DConvolution called")
end

Derivative[FConvolution] = DConvolution
Inplace[FConvolution]   = FConvolution
export FConvolution
Convolution(tensor::ADTensor,filter::ADTensor)=ADFunction(FConvolution,tensor filter)
export Convolution

