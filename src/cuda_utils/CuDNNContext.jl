type CuDNNContext{T}
handle::cudnnHandle_t
srcTensorDesc::cudnnTensorDescriptor_t
dstTensorDesc::cudnnTensorDescriptor_t
tensorFormat::Int
srcData::CuPtr
dstData::CuPtr
dataType::Type{T}

CuDNNContext{T}(tFormat::Int,dType::Type{T}) = begin
handle = CuDNN.cudnnCreate()
dataType = dType
srcTensorDesc = CuDNN.cudnnCreateTensorDescriptor()
dstTensorDesc = CuDNN.cudnnCreateTensorDescriptor()
tensorFormat =tFormat
srcData = CUDA.CuPtr()
dstData = CUDA.CuPtr()
end
end

export CuDNNContext

function free(ctx::CuDNNContext)
CuDNN.cudnnDestroyTensorDescriptor(ctx.srcTensorDesc)
CuDNN.cudnnDestroyTensorDescriptor(ctx.dstTensorDesc)
CuDNN.cudnnDestroy(ctx.handle)
CUDA.free(ctx.srcData)
CUDA.free(ctx.dstData)
end



