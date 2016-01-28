#cudnnActivationMode_t
const CUDNN_ACTIVATION_SIGMOID = 0
const CUDNN_ACTIVATION_RELU = 1
const CUDNN_ACTIVATION_TANH = 2
function cudnnActivationForward{T<:AbstractFloat}(handle::cudnnHandle_t,mode::Int,alpha::T,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,beta::T,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnActivationForward,(cudnnHandle_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,T[alpha],srcDesc,srcData,T[beta],destDesc,destData)
end

function cudnnActivationBackward{T<:AbstractFloat}(handle::cudnnHandle_t,mode::Int,alpha::T,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::CudaPtr,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr,beta::T,destDiffDesc::cudnnTensorDescriptor_t,destDiffData::CudaPtr)
@cudnncheck(:cudnnActivationBackward,(cudnnHandle_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,T[alpha],srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,T[beta],destDiffDesc,destDiffData)
end