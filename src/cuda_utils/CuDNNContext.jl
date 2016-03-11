type CudaTensor
handle::cudnnHandle_t
data::CudaArray
dimen::Int
dims::Array{Cint,1} #n,c,h,w
#stride::Array{Cint,1}
tensorDesc::cudnnTensorDescriptor_t
tensorFormat::Int
dataType::Int
operation # hold the CuDNN operation type
#Constructor for NDTensor
CudaTensor(h::cudnnHandle_t,data::CudaArray,dims::Array{Int,1},stride::Array{Int,1},dType::Int,operation) =begin
tensorDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(tensorDesc,dType,dims[1],dims[2],dims[3],dims[4])
tensor  = new(h,data,length(dims),dims,tensorDesc,1,dType,operation)
return tensor
end

end

type CudaFilter
data::CudaArray
#dimen::Int 
dims::Array{Int,1}
dataType::Int
filterDesc::cudnnFilterDescriptor_t
CudaFilter(data::CudaArray,dims::Array{Int,1},dataType) = begin
filterDesc = cudnnCreateFilterDescriptor()
cudnnSetFilter4dDescriptor(filterDesc,dataType,dims[1],dims[2],dims[3],dims[4])
filter = new(data,dims,dataType,filterDesc)
end
end


NDTensor(h,data,dims,stride,dtype) = CudaTensor(h,data,dims,stride,dtype,nothing)
export NDTensor
NDFilter(data,dims,dtype) = CudaFilter(data,dims,dtype)
export NDFilter


type ConvNode 
convDesc::cudnnConvolutionDescriptor_t
filterDesc::cudnnFilterDescriptor_t
biasDesc::cudnnTensorDescriptor_t
algo::Int
size::Int
end
export ConvNode


type PoolingNode
poolingDesc::cudnnPoolingDescriptor_t
size::Int
end
export PoolingNode

type LinearRectifierNode
normDesc::cudnnLRNDescriptor_t
size::Int
end
export LinearRectifierNode

function free(ctx::CudaTensor)
cudnnDestroyTensorDescriptor(ctx.srcTensorDesc)
cudnnDestroyTensorDescriptor(ctx.dstTensorDesc)
end

export free

