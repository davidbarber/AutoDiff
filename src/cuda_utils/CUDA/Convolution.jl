typealias cudnnConvolutionDescriptor_t Ptr{Void} # hold the description of a convolution operation
export cudnnConvolutionDescriptor_t
#cudnnConvolutionMode_t
const CUDNN_CONVOLUTION = 0
const CUDNN_CROSS_CORRELATION = 1

function checkConvolutionMode(mode::Int)
if mode != CUDNN_CONVOLUTION && mode != CUDNN_CROSS_CORRELATION
  error("unkonwn mode for convolution setting, expect 0 for convolution, 1 for cross correlation")
end
end

###### convolution descriptors ########
function cudnnCreateConvolutionDescriptor()
convDesc = cudnnConvolutionDescriptor_t[0]
@cudnncheck(:cudnnCreateConvolutionDescriptor,(Ptr{cudnnConvolutionDescriptor_t},),convDesc)
return convDesc[1]
end

function cudnnSetConvolution2dDescriptor(convDesc::cudnnConvolutionDescriptor_t,pad_h::Int,pad_w::Int,u::Int,v::Int,upscalex::Int,upscaley::Int,mode::Int)
checkConvolutionMode(mode)
@cudnncheck(:cudnnSetConvolution2dDescriptor,(cudnnConvolutionDescriptor_t,Cint,Cint,Cint,Cint,Cint,Cint,Cint),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
end

function cudnnGetConvolution2dDescriptor(convDesc::cudnnConvolutionDescriptor_t)
pad_h = Cint[0]
pad_w = Cint[0]
u = Cint[0]
v = Cint[0]
upscalex = Cint[0]
upscaley = Cint[0]
mode = Cint[0]
@cudnncheck(:cudnnGetConvolution2dDescriptor,(convDesc::cudnnConvolutionDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),convDesc,pad_h,pad_w,u,v,upscalex,upscaley,mode)
return (convDesc,(pad_h[1],pad_w[1]),(u[1],v[1]),(upscalex[1],upscaley[1]),mode[1])
end

function cudnnGetConvolution2dForwardOutputDim(convDesc::cudnnConvolutionDescriptor_t,inputTensorDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t)
n = Cint[0]
c = Cint[0]
h = Cint[0]
w = Cint[0]
@cudnncheck(:cudnnGetConvolution2dForwardOutputDim,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,n,c,h,w)
return (n[1],c[1],h[1],w[1])
end

#TODO: add the old version
function cudnnSetConvolutionNdDescriptor{T<:AbstractFloat}(convDesc::cudnnConvolutionDescriptor_t,arrayLength::Int,PadA::Array{Int,1},filterStrideA::Array{Int,1},upscaleA::Array{Int,1},mode::Int,dataType::Type{T})
checkConvolutionMode(Mode)
dtype = cudnnDataTypeCheck(dataType)
@cudnncheck(:cudnnSetConvolutionNdDescriptor_v3,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint),convDesc,arrayLength,padA,filterStrideA,mode,dtype)
end

#TODO: add the old version
function cudnnGetConvolutionNdDescriptor(convDesc::cudnnConvolutionDescriptor_t,arrayLengthRequested::Int,PadA::Array{Int,1},filterStrideA::Array{Int,1})
arrayLength = Cint[0]
mode = Cint[0]
dataType= Cint[0]
@cudnncheck(:cudnnGetConvolutionNdDescriptor,(cudnnConvolutionDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),convDesc,arrayLengthRequested,arrayLength,padA,filterStrideA,upscaleA,mode,dataType)
dtype = cudnnDataTypeConvert(dataType[1])
return(convDesc,arrayLength[1],PadA,filterStrideA,upscaleA,mode,dtype)
end

function cudnnGetConvolutionNdForwardOutputDim(convDesc::cudnnConvolutionDescriptor_t,inputTensorDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t,nbDims::Int)
tensorOutputDimA = Cint[nbDims]
@cudnncheck(:cudnnGetConvolutionNdForwardOutputDim,(cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Cint}),convDesc,inputTensorDesc,filterDesc,nbDims,tensorOutputDimA)
return tensorOutputDimA
end

function cudnnDestroyConvolutionDescriptor(convDesc::cudnnConvolutionDescriptor_t)
@cudnncheck(:cudnnDestroyConvolutionDescriptor,(cudnnConvolutionDescriptor_t,),convDesc)
end

########### forward convolution ################
#cudnnConvolutionFwdPreference_t
const CUDNN_CONVOLUTION_FWD_NO_WORKSAPCE = 0
const CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
const CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 3

#cudnnConvolutionFwdAlgo_t
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_RECOMP_GEMM = 1
const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
const CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4

#cudnnConvolutionFwdAlgoPerf_t
type cudnnConvolutionFwdAlgoPerf_t
 algo::Cint # refers to cudnnConvolutionFwdAlgo_t
 status::Cint # refers to cudnnStatus_t
 time::Cfloat 
 memory::Csize_t
end
export cudnnConvolutionFwdAlgoPerf_t

function cudnnFindConvolutionForwardAlgorithm(hanlde::cudnnHandle_t,srcDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,destDesc::cudnnTensorDescriptor_t,requestedAlgoCount::Int)
returnedAlgoCount = Cint[0]
perfResults = cudnnConvolutionFwdAlgoPerf_t[0]
@cudnncheck(:cudnnFindConvolutionForwardAlgorithm,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionFwdAlgoPerf_t}),handle,srcDesc,convDesc,destDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
return (returnedAlgoCount[1],perfResults[1])
end

function cudnnGetConvolutionForwardAlgorithm(handle::cudnnHandle_t,srcDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,destDesc::cudnnTensorDescriptor_t,preference::Int,memoryLimiteInBytes::UInt)
algo = Cint[0]
@cudnncheck(:cudnnGetConvolutionForwardAlgorithm,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Csize_t,Ptr{Cint}),handle,srcDesc,filterDesc,convDesc,destDesc,preference,memoryLimiteInBytes,algo)
return algo[1]
end

function cudnnGetConvolutionForwardWorkspaceSize(handle::cudnnHandle_t,srcDesc::cudnnTensorDescriptor_t,filterDesc::cudnnFilterDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,destDesc::cudnnTensorDescriptor_t,algo::Int)
sizeInBytes = Csize_t[0]
@cudnncheck(:cudnnGetConvolutionForwardWorkspacesize,(cudnnhandle_t,cudnnTensorDescriptor_t,cudnnFilterDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Csize_t}),handle,srcDesc,filterDesc,convDesc,destDesc,algo,sizeInBytes)
return sizeInBytes[1]
end

function cudnnConvolutionForward(handle::cudnnHandle_t,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,filterDesc::cudnnFilterDescriptor_t,filterData::CudaPtr,convDesc::cudnnConvolutionDescriptor_t,algo::Int,workSpace::CudaPtr,workSpaceSizeInBytes::UInt,beta,DestDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnConvolutionForward,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Cint,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData.p,filterDesc,filterData.p,convDesc,algo,workSpace.p,workSpaceSizeInBytes,beta,destDesc,destData)
return destData
end


##########Convolution Backword

#cudnnConvolutionBwdFilterPreference_t
const CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
const CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FRASTEST = 1
const CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2

#cudnnConvolutionBwdFilterAlgo_t
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3
const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 4

#cudnnConvolutionBwdFilterAlgoPerf_t
type cudnnConvolutionBwdFilterAlgoPerf_t
algo::Cint
status::Cint
time::Cfloat
memory::Csize_t
end

export cudnnConvolutionBwdFilterAlgoPerf_t

#cudnnConvolutionBwdDataPreference_t
CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2

#cudnnConvolutionBwdDataAlgo_t
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
const CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 4


#cudnnConvolutionBwdDataPerf_t
type cudnnConvolutionBwdDataAlgoPerf_t
algo::Cint
status::Cint
time::Cfloat
memory::Csize_t
end

export cudnnConvolutionBwdDataAlgoPerf_t

function cudnnConvolutionBackwardBias(handle::cudnnHandle_t,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,beta,destDesc::cudnnTensorDescriptor_t,destData::CudaPtr)
@cudnncheck(:cudnnConvolutionBackwardBias,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData,beta,destDesc,destData)
end

function cudnnFindConvolutionBackwardFilterAlgorithm(handle::cudnnHandle_t,srcDesc::cudnnTensorDescriptor_t,diffDesc::cudnnTensorDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,gradDesc::cudnnFilterDescriptor_t,requestedAlgoCount::Int)
returnedAlgoCount = Cint[0]
perfResults = cudnnConvolutionBwdFilterAlgoPerf_t[0]
@cudnncheck(:cudnnFindConvolutionBackwardFilterAlgorithm,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,Ptr{Cint},Ptr{cudnnConvolutionBackwardFilter}),hanlde,srcDesc,diffDesc,convDesc,gradDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
return (returnedAlgoCount[1],perfResults[1])
end

function cudnnGetConvolutionBackwardFilterAlgorithm(handle::cudnnHandle_t,srcDesc::cudnnTensorDescriptor_t,diffDesc::cudnnTensorDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,gradDesc::cudnnFilterDescriptor_t,preference::Int,memoryLimitInbytes::UInt)
algo = Cint[0]
@cudnncheck(:cudnnGetConvolutionBackwardFilterAlgorithm,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,Cint,Csize_t,Ptr{Cint}),handle,srcDesc,diffDesc,convDesc,gradDesc,preference,memoryLimitInbytes,algo)
return algo[1]
end

function cudnnGetConvolutionBackwardFilterWorkspaceSize(handle::cudnnHandle_t,srcDesc::cudnnTensorDescriptor_t,diffDesc::cudnnTensorDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,gradDesc::cudnnFilterDescriptor_t,algo::Int)
sizeInBytes = Csize_t[0]
@cudnncheck(:cudnnGetConvolutionBackwardFilterWorkspaceSize,(cudnnHandle_t,cudnnTensorDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnFilterDescriptor_t,Cint,Ptr{Csize_t}),handle,srcDesc,diffDesc,convDesc,gradDesc,algo,sizeInBytes)
return sizeInBytes[1]
end


# Add case check for old version
function cudnnConvolutionBackwardFilter(handle::cudnnHandle_t,alpha,srcDesc::cudnnTensorDescriptor_t,srcData::CudaPtr,diffDesc::cudnnTensorDescriptor_t,diffData::CudaPtr,convDesc::cudnnConvolutionDescriptor_t,algo::Int,workspace::CudaPtr,workspaceSizeInBytes::UInt,beta,gradDesc::cudnnFilterDescriptor_t,gradData::CudaPtr)
@cudnncheck(:cudnnConvolutionBackwardFilter_v3,(cudnnHandle_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Cint,Ptr{Void},Csize_t,cudnnFilterDescriptor_t,Ptr{Void}),handle,alpha,srcDesc,srcData.p,diffDesc,diffData.p,convDesc,workspaceSizeInBytes,workspace.p,beta,gradDesc,gradData.p)
end
#Add case check for old version
function cudnnConvolutionBackwardData(handle::cudnnHandle_t,alpha,filterDesc::cudnnFilterDescriptor_t,filterData::CudaPtr,diffDesc::cudnnTensorDescriptor_t,diffData::CudaPtr,convDesc::cudnnConvolutionDescriptor_t,algo::Int,workspace::CudaPtr,workspaceSizeInBytes::UInt,beta,gradDesc::cudnnTensorDescriptor_t,gradData::CudaPtr)
@cudnncheck(:cudnnConvolutionBackwardData_v3,(cudnnHandle_t,Ptr{Void},cudnnFilterDescriptor_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void},cudnnConvolutionDescriptor_t,Cint,Ptr{Void},Csize_t,Ptr{Void},cudnnTensorDescriptor_t,Ptr{Void}),handle,alpha,filterDesc,filterData,diffDesc,diffData,convDesc,algo,workspace,workspaceSizeInBytes,beta,gradDesc,gradData)
end

function cudnnFindConvolutionBackwardDataAlgorithm(handle::cudnnHandle_t,filterDesc::cudnnFilterDescriptor_t,diffDesc::cudnnTensorDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,gradDesc::cudnnTensorDescriptor_t,requestedAlgoCount::Int)
requestedAlgoCount = Cint[0]
perfResults = cudnnConvolutionBwdFilterAlgoPerf_t[0]
@cudnncheck(:cudnnFindConvolutionBackwardDataAlgorithm,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Cint},Ptr{cudnnConvolutionBwdFilterAlgoPerf_t}),handle,filterDesc,diffDesc,convDesc,gradDesc,requestedAlgoCount,returnedAlgoCount,perfResults)
return (requestedAlgoCount[1],perfResults[1])
end

function cudnnGetConvolutionbackwardDataAlgorithm(handle::cudnnHandle_t,filterDesc::cudnnFilterDescriptor_t,diffDesc::cudnnTensorDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,gradDesc::cudnnTensorDescriptor_t,preference::Int,memoryLimitInbytes::UInt)
algo = Cint[0]
@cudnncheck(:cudnnGetConvolutionbackwardDataAlgorithm,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Csize_t,Ptr{Cint}),handle,filterDesc,diffDesc,convDesc,gradDesc,preference,memoryLimitInbytes,algo)
return algo[1]
end

function cudnnGetConvolutionBackwardDataWorkspaceSize(handle::cudnnHandle_t,filterDesc::cudnnFilterDescriptor_t,diffDesc::cudnnTensorDescriptor_t,convDesc::cudnnConvolutionDescriptor_t,gradDesc::cudnnTensorDescriptor_t,algo::Int)
sizeInBytes = Csize_t[0]
@cudnncheck(:cudnnGetConvolutionBackwardDataWorkspaceSize,(cudnnHandle_t,cudnnFilterDescriptor_t,cudnnTensorDescriptor_t,cudnnConvolutionDescriptor_t,cudnnTensorDescriptor_t,Cint,Ptr{Csize_t}),handle,filterDesc,diffDesc,convDesc,gradDesc,algo,sizeInBytes)
return sizeInBytes[1]
end
