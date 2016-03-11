# Warning: in CuDNN v3.0 library guide, there is no mention of type cudnnLRNDescriptor_t, 
# however this type is used in all the method realated to LRN.
# So cudnnLRNDescriptor_t is declared here to macth the CuDNN method

typealias cudnnLRNDescriptor_t Ptr{Void} 
export cudnnLRNDescriptor_t
#cudnnLRNMode_t
const CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0

function cudnnCreateLRNDescriptor()
lrnDesc = cudnnLRNDescriptor_t[0]
@cudnncheck(:cudnnCreateLRNDescriptor,(cudnnLRNDescriptor_t,),lrnDesc)
return lrnDesc[1]
end

function cudnnSetLRNDescriptor(lrnDesc::cudnnLRNDescriptor_t,lrnN::Int,lrnAlpha::Float64,lrnBeta::Float64,lrnK::Float64)
@cudnncheck(:cudnnSetLRNDescriptor,(cudnnLRNDescriptor_t,Cuint,Cdouble,Cdouble,Cdouble),lrnDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
end

function cudnnGetLRNDescriptor(lrnDesc::cudnnLRNDescriptor_t)
lrnN = Cuint[0]
lrnAlpha = Cdouble[0]
lrnBeta = Cdouble[0]
lrnK = Cdouble[0]
@cudnncheck(:cudnnGetLRNDescriptor,(cudnnLRNDescriptor_t,Cuint,Cdouble,Cdouble,Cdouble),lrnDesc,lrnN,lrnAlpha,lrnBeta,lrnK)
return (lrnN[1],lrnAlpha[1],lrnBeta[1],lrnK[1])
end

function cudnnDestroyLRNDescriptor(lrnDesc::cudnnLRNDescriptor_t)
@cudnncheck(:cudnnDestroyLRNDescriptor,(cudnnLRNDescriptor_t,),lrnDesc)
end


function cudnnLRNCrossChanelForward{T<:AbstractFloat}(handle::cudnnHandle_t,lrnDesc::cudnnLRNDescriptor_t,mode::Int,alpha::T,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,beta::T,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnLRNCrossChannelForward,(cudnnHandle_t,cudnnLRNDescriptor_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,lrnDesc,mode,T[alpha],srcDesc,srcData,T[beta],destDesc,destData)
end

function cudnnLRNCrossChannelBackward{T<:AbstractFloat}(handle::cudnnHandle_t,lrnDesc::cudnnLRNDescriptor_t,mode::Int,alpha::T,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,srcDiffDesc::cudnnTensorDescriptor_t,srcDiffData::CudaPtr,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr,beta::T,destDiffDesc::cudnnTensorDescriptor_t,destDiffData::CudaPtr)
@cudnncheck(:cudnnLRNCrossChannelBackward,(cudnnHandle_t,cudnnLRNDescriptor_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,lrnDesc,mode,T[alpha],srcDesc,srcData,srcDiffDesc,srcDiffData,destDesc,destData,T[beta],destDiffDesc,destDiffData)
end

CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0
function cudnnDivisiveNormalizationForward(handle::cudnnHandle_t,normDesc::cudnnLRNDescriptor_t,mode::Int,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,srcMeansData::CudaPtr,tempData::CudaPtr,tempData2::CudaPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnDivisiveNormalizationForward,(cudnnHandle_t,cudnnLRNDescriptor_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,normDesc,mode,alpha,srcDesc,srcData,srcMeansData,tempData,tempdata2,beta,destDesc,destData)
end

function cudnnDivisiveNormalizationBackward(handle::cudnnHandle_t,normDesc::cudnnLRNDescriptor_t,mode::Int,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,srcMeansData::CudaPtr,tempData::CudaPtr,tempData2::CudaPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr,destDiffMean::CudaPtr)
@cudnncheck(:cudnnDivisiveNormalizationForward,(cudnnHandle_t,cudnnLRNDescriptor_t,Cint,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void}),handle,normDesc,mode,alpha,srcDesc,srcData,srcMeansData,tempData,tempdata2,beta,destDesc,destData,destDiffMean)
end
