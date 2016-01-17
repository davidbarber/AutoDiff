type CuDNNContext{T}
handle::cudnnHandle_t
srcTensorDesc::cudnnTensorDescriptor_t
dstTensorDesc::cudnnTensorDescriptor_t
tensorFormat::Int
dataType::Type{T}

CuDNNContext{T}(h::cudnnHandle_t,tFormat::Int,dType::Type{T}) = begin
srcTensorDesc = CuDNN.cudnnCreateTensorDescriptor()
dstTensorDesc = CuDNN.cudnnCreateTensorDescriptor()
context  = new(h,srcTensorDesc,dstTensorDesc,tFormat,dType)
return context
end
end

export CuDNNContext


function free(ctx::CuDNNContext)
cudnnDestroyTensorDescriptor(ctx.srcTensorDesc)
cudnnDestroyTensorDescriptor(ctx.dstTensorDesc)
end
export free
global handle 
function createHandle()
global handle 
handle =  cudnnCreate()
end
export handle
export createHandle
function freehandle()
global handle
cudnnDestroy(handle)
end
export freehandle

