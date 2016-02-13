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

if PROC=="GPU"
function FConvolution(handle,mapping::NTuple{2,Int},value::CudaArray,auxvalue,t::CudaArray,f::CudaArray)
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

function DConvolution(handle,mapping::NTuple{2,Int},derivativeIDX,f_c,faux_c,grad_c,grad_n,t::CudaArray,f::CudaArray)
# grad_n child
# grad_c current 
alpha = 1.0
beta =0.0
convDesc = cudnnCreateConvolutionDescriptor()
cudnnSetConvolution2dDescriptor(convDesc,0,0,1,1,1,1,0)
diffDataDesc = cudnnCreateTensorDescriptor()
dtype = eltype(t)
dataType = cudnnDataTypeCheck(dtype)
(n,c,h,w)= size(grad_c)
cudnnSetTensor4dDescriptor(diffDataDesc,dataType,n,c,h,w)


srcDataDesc = cudnnCreateTensorDescriptor()
(n,c,h,w) = size(t)
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)

(i,o) = mapping 
filterDesc = cudnnCreateFilterDescriptor()
(h,w) = size(f)
cudnnSetFilter4dDescriptor(filterDesc,dataType,i,o,h,w)
workspace = CUDA_NULL
if derivativeIDX ==1
dtype = eltype(t)
temp = CudaArray(dtype,size(t))
cudnnConvolutionBackwardData(handle,alpha,filterDesc,f.ptr,diffDataDesc,grad_c.ptr,convDesc,0,workspace,0,beta,srcDataDesc,temp.ptr)
CUBLAS.axpy!(1.0,temp,grad_n)
free(temp)

elseif derivativeIDX ==2
dtype = eltype(t)
temp = CudaArray(dtype,size(t))
cudnnConvolutionBackwardFilter(handle,alpha,srcDataDesc,t.ptr,diffDataDesc,grad_c.ptr,convDesc,0,workspace,0,beta,filterDesc,temp.ptr)

CUBLAS.axpy!(1.0,temp,grad_n)
free(temp)

end
cudnnDestroyTensorDescriptor(srcDataDesc)
cudnnDestroyTensorDescriptor(diffDataDesc)
cudnnDestroyFilterDescriptor(filterDesc)
cudnnDestroyConvolutionDescriptor(convDesc)
free(workspace)
return grad_n
end
end

Derivative[FConvolution] = DConvolution
Inplace[FConvolution]   = FConvolution
export FConvolution
Convolution(tensor::ADnode,filters::ADnode,mapping::NTuple{2,Int})=ADFunction(FConvolution,tensor,filters;special=mapping)
export Convolution

