#cudnnSoftmaxAlgorithm_t
const CUDNN_SOFTMAX_FAST = 0
const CUDNN_SOFTMAX_ACCURATE = 1
const CUDNN_SOFTMAX_LOG = 2

#cudnnSoftmaxMode_t
const CUDNN_SOFTMAX_MODE_INSTANCE = 0
const CUDNN_SOFTMAX_MODE_CHANNEL = 1


function cudnnSoftmaxForward{T<:AbstractFloat}(handle::cudnnHandle_t,algorithm::Int,mode::Int,alpha::T,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,beta::T,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnSoftmaxForward,(cudnnHandle_t,Cint,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algorithm,mode,T[alpha],srcDesc,srcData,T[beta],destDesc,destData)
end

function cudnnSoftmaxBackward{T<:AbstractFloat}(handle::cudnnHandle_t,algorithm::Int,mode::Int,alpha::T,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::CudaPtr,beta::T,destDiffDesc::cudnnTensorDescriptor_t,destDiffData::CudaPtr)
@cudnncheck(:cudnnSoftmaxBackward,(cudnnHandle_t,Cint,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algorithm,mode,T[alpha],srcDesc,srcData,srcDiffDesc,srcDiffData,T[beta],destDiffDesc,destDiffData)
end