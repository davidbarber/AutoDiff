
include("CUDA/CuDNN.jl")

using CuDNN
using CUDA
include("CuDNNContext.jl")
abstract Node

type ConvNode <: Node
convDesc::cudnnConvolutionDescriptor_t
filterDesc::cudnnFilterDescriptor_t
biasDesc::cudnnTensorDescriptor_t
algo::Int
size::Int
end



function convolutionSetup(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int,inputs::Int,outputs::Int,kDim::Int)
CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
const dims = 4
tensorOutputDim = [n,c,h,w]
filterDim = [outputs,inputs,kDim,kDim]
filterDesc = CuDNN.cudnnCreateFilterDescriptor()
CuDNN.cudnnSetFilterNdDescriptor(filterDesc,net.dataType,dims,filterDim)
const convDims = 2;
padA= [0,0];
filterStrideA=[1,1];
upscaleA = [1,1];
CUDNN_CROSS_CORRELATION = 1;
CuDNN.cudnnSetConvolutionNdDescriptor(node.convDesc,convDims,padA,filterStrideA,upscaleA,CUDNN_CROSS_CORRELATION,net.dataType)
tensorOutputDimA = CuDNN.cudnnGetConvolutionNdForwardOutputDim(node.convDesc,net.srcTensorDesc,node.filterDesc,dims,tensorOutputDim)
n=tensorOutputDimA(1)
c=tensorOutputDimA(2)
h=tensorOutputDimA(3)
w=tensorOutputDimA(4)
convDesc = CuDNN.cudnnCreateConvolutionDescriptor()
CuDNN.cudnnSetTensorNdDescriptor(net.dstTensorDesc,net.tensorFormat,net.dataType,n,c,h,w)
algo = CuDNN.cudnnGetConvolutionForwardAlgorithm(ctx.handle,ctx.srcTensorDesc,ctx.filterDesc,convDesc,ctx.dstTensorDesc,1,0)

biasDesc = CuDNN.cudnnCreateTensorDescriptor()
CuDNN.cudnnSetTensorNdDescriptor(biasDesc,ctx.tensorFormat,ctx.dataType,1,c,1,1)
return ConvNode(convDesc,filterDesc,biasDesc,algo,n*h*c*w)
end

#TODO
function forward(ctx::CuDNNContext,node::ConvNode)
resize(ctx,node.size)
alpha =1.0
beta = 0.0
sizeInByte = CuDNN.cudnnGetConvolutionForwardWorkspaceSize(ctx.handle,ctx.srcTensorDesc,filterDesc,convDesc,ctx.dstTensorDesc,algo)
workspace = nothing
if(sizeInByte ==0)
workspace = CUDA.CuPtr()
else
workspace = CUDA.cualloc(UInt8,workspaceSize)
end
#CuDNN.cudnnConvolutionForward(ctx.handle,alpha,ctx.srcTensorDesc,ctx.srcData,node.filterDesc,)
end




#TODO:wrap backward
function backward()

end

function free()

end


type PoolingNode <:Node
poolingDesc::cudnnPoolingDescriptor_t
#TODO:: might need two fields below
#srcDesc::cudnnTensorDescriptor_t
#dstDesc::cudnnTensorDescriptor_t
size::Int
end

function poolingSetup(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int)
const dims = 2
windowDim = [2,2]
padding = [0,0]
stride = [2,2]
poolingDesc = cudnnCreatePoolingDescriptor()
CuDNN.cudnnSetPoolingNdDescriptor(poolingDesc,0,dims,windowDim,padding,stride)

CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)

const tensorDims = 4
tensorOutPutDims =[n,c,h,w]
outputDim=CuDNN.cudnnGetPoolingNdForwardOutputDim(poolingDesc,ctx.srcTensorDesc,tensorDims,tensorOuputDims)
n=outputDim[1]
c=outputDim[2]
h=outputDim[3]
w=outputDim[4]

CuDNN.cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
node = PoolingNode(poolingDesc,n*c*h*w)
end

function forward(ctx::CuDNNContext,node::PoolingNode,inputs)
alpha = 1.0
beta = 0.0
resize(ctx,node.size)
CuDNN.cudnnPoolingForward(ctx.handle,node.poolingDesc,alpha,ctx.srcTensorDesc,inputs,beta,ctx.destDesc,ctx.dstData)
end

#TODO: wrap backward
function backward()

end

function free()

end


type LinearRectifierNode <:Node
normDesc::cudnnLRNDescriptor_t
size::Int
end

function linearRectifierSetup(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int)
normDesc = CuDNN.cudnnCreateLRNDescriptor()
N = 5
alpha = 0.0001
beta = 0.75
K = 1.0
CuDNN.cudnnSetLRNDescriptor(normDesc,N,alpha,beta,K)
CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
CuDNN.cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
return LinearRectifierNode(normDesc,n*c*h*w)
end

function forward(ctx::CuDNNContext,node::LinearRectifierNode,inputs)
alpha = 1.0
beta = 0.0
resize(ctx,node.size)
CuDNN.cudnnLRNCrossChanelForward(ctx.cudnnHandle,node.normDesc,0,alpha,ctx.srcTensorDesc,inputs,beta,ctx.destDesc,ctx.dstData)
end

#TODO: wrap backward
function backward()

end

function free()

end


#softmax
function softmaxForward(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int)
resize(ctx,n*w*c*h)
CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
CuDNN.cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
alpha = 1.0
beta = 0.0
CuDNN.cudnnSoftmaxForward(ctx.handle,1,1,alpha,ctx.srcTensorDesc,ctx.srcData,beta,ctx.dstTensorDesc,ctx.dstData)
end

#TODO: wrapping this function
function softmaxBackward()

end

#activation
function activationForward(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int)
resize(ctx,n*c*h*w)
CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
alpha = 1.0
beta = 0.0
CuDNN.cudnnActivationForward(ctx.handle,1,alpha,ctx.srcTensorDesc,ctx.srcData,beta,ctx.dstTensorDesc,ctx.dstData)
end

function activationBackward()

end


function resize(ctx::CuDNNContext,size::Int)
if(data.p != nothing)
CUDA.free(ctx.dstData)
end
CUDA.cualloc(ctx.dataType,size)
end
