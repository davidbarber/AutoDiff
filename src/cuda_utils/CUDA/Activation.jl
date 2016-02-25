#cudnnActivationMode_t
const CUDNN_ACTIVATION_SIGMOID = 0
const CUDNN_ACTIVATION_RELU = 1
const CUDNN_ACTIVATION_TANH = 2
function cudnnActivationForward(handle::cudnnHandle_t,mode::Int,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnActivationForward,(cudnnHandle_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,Float64[alpha],srcDesc,srcData,Float64[beta],destDesc,destData)
end

function cudnnActivationBackward(handle::cudnnHandle_t,mode::Int,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::CudaPtr,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr,beta,destDiffDesc::cudnnTensorDescriptor_t,destDiffData::CudaPtr)
@cudnncheck(:cudnnActivationBackward,(cudnnHandle_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,srcDesc,srcData.p,srcDiffDesc,srcDiffData.p,destDesc,destData.p,beta,destDiffDesc,destDiffDesc.p)
end