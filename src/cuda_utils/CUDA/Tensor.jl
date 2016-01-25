# Tensor discriptor 
typealias cudnnTensorDescriptor_t Ptr{Void} # hold the description of generic n-D dataset 
export cudnnTensorDescriptor_t
#cudnnTensorFormat_t
const CUDNN_TENSOR_NCHW = 0 #data laid out order: bancth, channel, rows, columns
const CUDNN_TENSOR_NHWC = 1 #data laid out order: bancth, rows, columns, channel

function cudnnCreateTensorDescriptor()
tensorDesc = cudnnTensorDescriptor_t[0]
@cudnncheck(:cudnnCreateTensorDescriptor,(Ptr{cudnnTensorDescriptor_t},),tensorDesc)
return tensorDesc[1]
end



function cudnnSetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t,dataType::Int,n,c,h,w)
@cudnncheck(:cudnnSetTensor4dDescriptor,(cudnnTensorDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint),tensorDesc,0,dataType,n,c,h,w)
end


function cudnnSetTensor4dDescriptorEx{T<:AbstractFloat}(tensorDesc::cudnnTensorDescriptor_t,dataType::Type{T},n,c,h,w,nStride,cStride,hStride,wStride)
@cudnncheck(:cudnnSetTensor4dDescriptorEx,(cudnnTensorDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),n,c,h,w,nStride,cStride,hStride,wStride)
end

function cudnnGetTensor4dDescriptor(tensorDesc::cudnnTensorDescriptor_t)
dataType = Cint[0]
n = Cint[0]
c = Cint[0]
h = Cint[0]
w = Cint[0]
nStride = Cint[0]
cStride = Cint[0]
hStride = Cint[0]
wStride = Cint[0]
@cudnncheck(:cudnnGetTensor4dDescriptor,(cudnnTensorDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ref{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride)
#dtype = cudnnDataTypeConvert(dataType[1])
return ((n[1],c[1],h[1],w[1]),(nStride[1],cStride[1],hStride[1],wStride[1]))
end


function cudnnSetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t,dataType,nbDims,dimA,strideA)
@cudnncheck(:cudnnSetTensorNdDescriptor,(cudnnTensorDescriptor_t,Clonglong,Clonglong,Ptr{Clonglong},Ptr{Clonglong}),tensorDesc,dataType,nbDims,dimA,strideA)
end

function cudnnGetTensorNdDescriptor(tensorDesc::cudnnTensorDescriptor_t,nbDimsRequested::UInt)
dataType = Cint[0]
nbDims = Cint[0]
dimA = Cint[0]
strideA = Cint[0]
@cudnncheck(:cudnnGetTensorNdDescriptor,(cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),tensorDesc,nbDimsRequested,datatype,nbDims,dimA,strideA)
dtype = cudnnDataTypeConvert(dataType[1])
return (tensorDesc,dtype,nbDims,dimA,strideA)
end

function cudnnDestroyTensorDescriptor(tensorDesc::cudnnTensorDescriptor_t)
@cudnncheck(:cudnnDestroyTensorDescriptor,(cudnnTensorDescriptor_t,),tensorDesc)
end


#WARN: alpha, beta should be float, but in CuDNN.h it is void
function cudnnTransformTensor(handle::cudnnHandle_t,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnTransformTensor,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData.p,beta,destDesc,destData.p)
end

#cudnnAddMode_t
const CUDNN_ADD_IMAGE = 0
const CUDNN_ADD_SAME_HW = 0
const CUDNN_ADD_FEATURE_MAP = 1
const CUDNN_ADD_SAM_CHW = 1
const CUDNN_ADD_SAME_C = 2
const CUDNN_ADD_FULL_TENSOR = 3

#TODO: if the version is less than v3.0 replace cudnnAddTensor_v3 by cudnnAddTensor
#TODO: add scaling_type(), to convert alpha, beta to scaling_type
#WARN: alpha, beta should be float, but in CuDNN.h it is void
function cudnnAddTensor(handle::cudnnHandle_t,alpha,biasDesc::cudnnTensorDescriptor_t,biasData::CudaPtr,beta,srcDestDesc::cudnnTensorDescriptor_t,srcDestData::CudaPtr)
@cudnncheck(:cudnnAddensor_v3,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,biasDesc,biasData.p,beta,srcDestDesc,srcDestData.p)
end

function cudnnSetTensor(handle::cudnnHandle_t,srcDestDesc::cudnnTensorDescriptor_t,srcDestData::CudaPtr,value)
@cudnncheck(:cudnnSetTensor,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,srcDestDesc,srcDestData.p,value)
end


function cudnnScaleTensor(handle::cudnnHandle_t,srcDestDesc::cudnnTensorDescriptor_t,srcDestData::CudaPtr,alpha)
@cudnncheck(:cudnnScaleTensor,(cudnnHandle_t,cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,srcDestDesc,srcDestData.p,alpha)
end

