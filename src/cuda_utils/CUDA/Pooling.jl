typealias cudnnPoolingDescriptor_t Ptr{Void}
export cudnnPoolingDescriptor_t
#cudnnPoolingMode_t
const CUDNN_POOLING_MAX = 0
const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2

function cudnnCreatePoolingDescriptor()
poolingDesc = cudnnPoolingDescriptor_t[0]
@cudnncheck(:cudnnCreatePoolingDescriptor,(cudnnPoolingDescriptor_t,),poolingDesc)
return poolingDesc[1]
end

function cudnnSetPooling2dDescriptor(poolingDesc::cudnnPoolingDescriptor_t,mode::Int,windowHeight::Int,windowWidth::Int,verticalPadding::Int,horizontalPadding::Int,verticalStride::Int,horizontalStride::Int)
@cudnncheck(:cudnnSetPooling2dDescriptor,(cudnnPoolingDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,Cint),poolingDesc,mode,windowHeight,windowWidth,verticalStride,horizontalPadding,verticalStride,horizontalStride)
end

function cudnnGetPooling2dDescriptor(poolingDesc::cudnnPoolingDescriptor_t)
mode = Cint[0]
windowHeight = Cint[0]
windowWidth = Cint[0]
verticalPadding = Cint[0]
horizontalPadding = Cint[0]
verticalStride = Cint[0]
horizontalStride = Cint[0]
@cudnncheck(:cudnnGetPooling2dDescriptor,(cudnnPoolingDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,mode,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride)
return (mode[1],(windowHeight[1],windowWidth[1]),(verticalPadding[1],horizontalPadding[1]),(verticalStride[1],horizontalStride[1]))
end

function cudnnSetPoolingNdDescriptor(poolingDesc::cudnnPoolingDescriptor_t,mode::Int,nbDims::Int,windowDimA::Array{Int,1},paddingA::Array{Int,1},strideA::Array{Int,1})
@cudnncheck(:cudnnSetPoolingNdDescriptor,(cudnnPoolingDescriptor_t,Cint,Cint,Ptr{Int},Ptr{Int},Ptr{Int}),poolingDesc,mode,nbDims,windowDimA,paddingA,strideA)
end

function cudnnGetPoolingNdDescriptor(poolingDesc::cudnnPoolingDescriptor_t,nbDimsRequested::Int)
mode = Cint[0]
nbDims = Cint[0]
windowDimA = Array(Cint,nbDimsRequested)
paddingA  = Array(Cint,nbDimsRequested)
strideA = Array(Cint,nbDimsRequested)
@cudnncheck(:cudnnGetPoolingNdDescriptor,(cudnnPoolingDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),cudnnPoolingDescriptor_t,nbDimsRequested,nbDims,windowDimA,paddingA,strideA)
return (mode[1],nbDims[1],windowDimA,paddingA,strideA)
end

function cudnnDestroyPoolingDescriptor(poolingDesc::cudnnPoolingDescriptor_t)
@cudnncheck(:cudnnDestroyPoolingDescriptor,(cudnnPoolingDescriptor_t,),poolingDesc)
end

function cudnnGetPooling2dForwardOutputDim(poolingDesc::cudnnPoolingDescriptor_t,inputDesc::cudnnTensorDescriptor_t)
outN = Cint[0]
outC = Cint[0]
outH = Cint[0]
outW = Cint[0]
@cudnncheck(:cudnnGetPooling2dForwardOutputDim,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),poolingDesc,inputDesc,outN,outC,outH,outW)
return (outN[1],outC[1],outH[1],outW[1])
end

function cudnnGetPoolingNdFowardOutputDim(poolingDesc::cudnnPoolingDescriptor_t,inputDesc::cudnnTensorDescriptor_t,nbDims::Int)
outDimA = Array(Cint,nbDims)
@cudnncheck(:cudnnGetPoolingNdFowardOuputDim,(cudnnPoolingDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint}),poolingDesc,inputDesc,nbDims,outDimA)
return outDimA
end


#WARN: alpha, beta should be float, but in CuDNN.h it is void
function cudnnPoolingForward(handle::cudnnHandle_t,poolingDesc::cudnnPoolingDescriptor_t,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnPoolingForward,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,Float64[alpha],srcDesc,srcData,Float64[beta],destDesc,destData)
end

function cudnnPoolingBackward(handle::cudnnHandle_t,poolingDesc::cudnnPoolingDescriptor_t,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,srcDiff::cudnnTensorDescriptor_t,srcDiffData::CudaPtr,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr,beta,destDiff::cudnnTensorDescriptor_t,destDiffData::CudaPtr)
@cudnncheck(:cudnnPoolingBackward,(cudnnHandle_t,cudnnPoolingDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,poolingDesc,alpha,srcDesc,srcData.p,srcDiff,srcDiffData.p,destDesc,destdata.p,beta,destdiff,destDiffData.p)
end
