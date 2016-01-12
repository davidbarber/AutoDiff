#cudnnActivationMode_t
const CUDNN_ACTIVATION_SIGMOID = 0
const CUDNN_ACTIVATION_RELU = 1
const CUDNN_ACTIVATION_TANH = 2
function cudnnActivationForward(handle::cudnnHandle_t,mode::Int,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CuPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CuPtr)
@cudnncheck(:cudnnActivationForward,(cudnnHandle_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,srcDesc,srcData.p,beta,destDesc,destData.p)
end

function cudnnActivationBackward(handle::cudnnHandle_t,mode::Int,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CuPtr,srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::CuPtr,destDesc::cudnnTensorDescriptor_t,destData::CuPtr,beta,destDiffDesc::cudnnTensorDescriptor_t,destDiffData::CuPtr)
@cudnncheck(:cudnnActivationBackward,(cudnnHandle_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,mode,alpha,srcDesc,srcData.p,srcDiffDesc,srcDiffData.p,destDesc,destData.p,beta,destDiffDesc,destDiffDesc.p)
end