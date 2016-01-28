#TODO this is the CPU version 
function FConvolution!(inputs::Array,filters::Array)
println("CPU version under develop")
return inputs, filters
end
#TODO this is the CPU version
function DConvolution()
println("CPU version under develop")
return 0
end


function FConvolution!(handle,mapping::NTuple{2,Int},value::CudaArray,auxvalue,t::CudaArray,f::CudaArray)
# Creation 
free(value)
(n,c,h,w) = size(t)
dtype = eltype(t)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)
(i,o) = mapping 
filterDesc = cudnnCreateFilterDescriptor()
(h,w) = size(f)
cudnnSetFilter4dDescriptor(filterDesc,dataType,i,o,h,w)

convDesc = cudnnCreateConvolutionDescriptor()
cudnnSetConvolution2dDescriptor(convDesc,0,0,1,1,1,1,0)
(n,c,h,w)= cudnnGetConvolution2dForwardOutputDim(convDesc,srcDataDesc,filterDesc)

value = CudaArray(dtype,n,c,h,w)
println(size(value))
dstDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDataDesc,dataType,n,c,h,w)
algo = cudnnGetConvolutionForwardAlgorithm(handle,srcDataDesc,filterDesc,convDesc,dstDataDesc,1,0)

alpha =1.0
beta = 0.0
sizeInByte = cudnnGetConvolutionForwardWorkspaceSize(handle,srcDataDesc,filterDesc,convDesc,dstDataDesc,algo)
workspace = nothing
if(sizeInByte ==0)
workspace = CudaPtr()
else
workspace =CUDArt.malloc(sizeInByte)
end
cudnnConvolutionForward(handle,alpha,srcDataDesc,t.ptr,filterDesc,f.ptr,convDesc,algo,workspace,sizeInByte,beta,dstDataDesc,value.ptr)
cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(dstDataDesc)
cudnnDestroyFilterDescriptor(filterDesc)
cudnnDestroyConvolutionDescriptor(convDesc)
free(workspace)

return value
end

function DConvolution(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray)
# grad_n child
# grad_c current 



if derivativeIDX ==1




elseif derivativeIDX ==2
cudnnGetConvolutionBackwardFilterWorkspaceSize()
cudnnConvolutionBackwardFilter()


end
end

Derivative[FConvolution!] = DConvolution
Inplace[FConvolution!]   = FConvolution!
export FConvolution
Convolution(tensor::ADnode,filters::ADnode,mapping::NTuple{2,Int})=ADFunction(FConvolution!,tensor,filters;special=mapping)
export Convolution

