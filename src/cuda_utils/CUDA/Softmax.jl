#cudnnSoftmaxAlgorithm_t
const CUDNN_SOFTMAX_FAST = 0
const CUDNN_SOFTMAX_ACCURATE = 1
const CUDNN_SOFTMAX_LOG = 2

#cudnnSoftmaxMode_t
const CUDNN_SOFTMAX_MODE_INSTANCE = 0
const CUDNN_SOFTMAX_MODE_CHANNEL = 1


function cudnnSoftmaxForward(handle::cudnnHandle_t,algorithm::Int,mode::Int,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnSoftmaxForward,(cudnnHandle_t,Cint,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algorithm,mode,Float64[alpha],srcDesc,srcData,Float64[beta],destDesc,destData)
end

function cudnnSoftmaxBackward(handle::cudnnHandle_t,algorithm::Int,mode::Int,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::CudaPtr,beta,destDiffDesc::cudnnTensorDescriptor_t,destDiffData::CudaPtr)
@cudnncheck(:cudnnSoftmaxBackward,(cudnnHandle_t,Cint,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,algorithm,mode,alpha,srcDesc,srcData.p,srcDiffDesc,srcDiffData.p,beta,destDiffDesc,destDiffData.p)
end