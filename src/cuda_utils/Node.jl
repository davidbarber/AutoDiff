
include("CUDA/CuDNN.jl")
include("CuDNNContext.jl")
abstract Node

type ConvNode <: Node
convDesc::cudnnConvolutionDescriptor_t
filterDesc::cudnnFilterDescriptor_t
biasDesc::cudnnTensorDescriptor_t
ctx::CuDNNContext
algo::Int
size::Int
end


type TensorDesc
n::Int #bancth 
c::Int #channel
h::Int #height
w::Int #weight
end


function convolutionSetup(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int,inputs::Int,outputs::Int,kDim::Int)
dType = cudnnDataTypeCheck(ctx.dataType)
cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
const dims = 4
tensorOutputDim = [n,c,h,w]
filterDim = [outputs,inputs,kDim,kDim]
filterDesc = cudnnCreateFilterDescriptor()
cudnnSetFilterNdDescriptor(filterDesc,ctx.dataType,dims,filterDim)
const convDims = 2;
padA= [0,0];
filterStrideA=[1,1];
upscaleA = [1,1];
CUDNN_CROSS_CORRELATION = 1;
convDesc = cudnnCreateConvolutionDescriptor()
cudnnSetConvolutionNdDescriptor(convDesc,convDims,padA,filterStrideA,upscaleA,CUDNN_CROSS_CORRELATION,ctx.dataType)
tensorOutputDimA = cudnnGetConvolutionNdForwardOutputDim(convDesc,ctx.srcTensorDesc,filterDesc,dims,tensorOutputDim)
n=tensorOutputDimA(1)
c=tensorOutputDimA(2)
h=tensorOutputDimA(3)
w=tensorOutputDimA(4)

cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,ctx.dataType,n,c,h,w)
algo = cudnnGetConvolutionForwardAlgorithm(ctx.handle,ctx.srcTensorDesc,ctx.filterDesc,convDesc,ctx.dstTensorDesc,1,0)

biasDesc = cudnnCreateTensorDescriptor()
cudnnSetTensorNdDescriptor(biasDesc,ctx.tensorFormat,ctx.dataType,1,c,1,1)
return ConvNode(convDesc,filterDesc,biasDesc,ctx,algo,n*h*c*w)
end

function forward(node::ConvNode,inputs::CudaArray,filters::CudaArray)

out = CudaArray()
alpha =1.0
beta = 0.0
sizeInByte = cudnnGetConvolutionForwardWorkspaceSize(node.ctx.handle,node.ctx.srcTensorDesc,node.filterDesc,node.convDesc,node.ctx.dstTensorDesc,algo)
workspace = nothing
if(sizeInByte ==0)
workspace = CudaPtr()
else
workspace =malloc(sizeInByte)
end
cudnnConvolutionForward(node.ctx.handle,alpha,node.ctx.srcTensorDesc,inputs,node.filterDesc,filters,node.convDesc,node.algo,workspace,sizeInByte,beta,node.destDesc,out)
return out
end

function backward(node::ConvNode,bias::CudaArray,filters::CudaArray,inputs::CudaArray)
alpha = 1.0
beta = 0.0
biasDiff = CudaArray()
cudnnConvolutionBackwardBias(node.ctx.handle,alpha,node.biasDesc,inputs,beta,node.ctx.dstTensorDesc,biasDiff)
filterDiff = CudaArray()
cudnnConvolutionBackwardFilter(node.ctx.handle,alpha,node.ctx.dstTensorDesc,filters,node.ctx.srcTensorDesc,biasDiff,node.convDesc,beta,node.filterDesc,filterDiff)
grad = CudaArray(ctx.dataType,node.size)
cudnnConvolutionBackwardData(node.ctx.handle,alpha,node.filterDesc,filters,ctx.dstTensorDesc,filterDiff,node.convDesc,0,C_NULL,0,beta,gradDesc::node.ctx.srcTensorDesc,grad)
return biasDiff,filterDiff,grad
end



# Function below still need to be modified, waiting for the confirmation from David
#TODO add bias
function biasAdd(ctx::CuDNNContext,node::ConvNode,inputs::CudaArray)



end


function free()

end


type PoolingNode <:Node
poolingDesc::cudnnPoolingDescriptor_t
size::Int
end

function poolingSetup(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int)
dType = cudnnDataTypeCheck(ctx.dataType)
const dims = 2
windowDim = [2,2]
padding = [0,0]
stride = [2,2]
poolingDesc = cudnnCreatePoolingDescriptor()
cudnnSetPoolingNdDescriptor(poolingDesc,0,dims,windowDim,padding,stride)

cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)

const tensorDims = 4
tensorOutPutDims =[n,c,h,w]
outputDim=cudnnGetPoolingNdForwardOutputDim(poolingDesc,ctx.srcTensorDesc,tensorDims,tensorOuputDims)
n=outputDim[1]
c=outputDim[2]
h=outputDim[3]
w=outputDim[4]

cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
return PoolingNode(poolingDesc,n*c*h*w)
end

function forward(ctx::CuDNNContext,node::PoolingNode,inputs::CudaArray)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,node.size)
cudnnPoolingForward(ctx.handle,node.poolingDesc,alpha,ctx.srcTensorDesc,inputs,beta,ctx.destDesc,out)
return out
end


function backward(ctx::CuDNNContext,node::PoolingNode,inputs::CudaArray,outputs::CudaArray,diffDesc::cudnnTensorDescriptor_t,diff::CudaArray)
alpha = 1.0
beta = 0.0 
grad = CudaArray(ctx.dataType,node.size)
cudnnPoolingBackward(ctx.handle,node.poolingDesc,alpha,ctx.dstTensorDesc,outputs,diffDesc,diff,ctx.srcTensorDesc,inputs,beta,ctx.srcTensorDesc,grad)
return grad
end

function free()

end


type LinearRectifierNode <:Node
normDesc::cudnnLRNDescriptor_t
size::Int
end

function linearRectifierSetup(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int)
dType = cudnnDataTypeCheck(ctx.dataType)
normDesc = CuDNN.cudnnCreateLRNDescriptor()
N = 5
alpha = 0.0001
beta = 0.75
K = 1.0
CuDNN.cudnnSetLRNDescriptor(normDesc,N,alpha,beta,K)
CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
CuDNN.cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
return LinearRectifierNode(normDesc,n*c*h*w)
end

#TODO still need to be changed to fit ADNode 
function forward(ctx::CuDNNContext,node::LinearRectifierNode,inputs::CudaArray)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,node.size)
cudnnLRNCrossChanelForward(ctx.cudnnHandle,node.normDesc,0,alpha,ctx.srcTensorDesc,inputs,beta,ctx.destDesc,out)
return out
end

#TODO: still need to be change to fit ADNode
function backward(ctx::CuDNNContext,node::LinearRectifierNode,inputs::CudaArray,outputs::CudaArray,diffDesc::cudnnTensorDescriptor_t,diff::CudaArray)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,node.size)
cudnnLRNCrossChanelBackward(ctx.handle,node.normDesc,0,alpha,ctx.dstTensorDesc,outputs,diffDesc,diff,ctx.srcTensorDesc,inputs,beta,ctx.srcTensorDesc,out)
return out
end

#TODO Normalised version should be wrap

function free()

end


#softmax
function softmaxForward(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int,inputs::CudaArray)
dType = cudnnDataTypeCheck(ctx.dataType)
out = CudaArray(ctx.dataType,n*c*h*w)
cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
alpha = 1.0
beta = 0.0
cudnnSoftmaxForward(ctx.handle,1,1,alpha,ctx.srcTensorDesc,inputs,beta,ctx.dstTensorDesc,out)
return out
end

#TODO: still need to be change to fit ADnode
function softmaxBackward(ctx::CuDNNContext,inputs::CudaArray,ouputs::CudaArray,diffDesc::cudnnTensorDescriptor_t,diff:CudaArray,n,w,h,c)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,n*w*h*c)
cudnnSoftmaxBackward(ctx.handle,1,1,alpha,ctx.dstTensorDesc,outputs,diffDesc,diff,beta,ctx.srcTensorDesc,out)
end

#activation
function activationForward(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int,inputs::CudaArray)
dType = cudnnDataTypeCheck(ctx.dataType)
out = CudaArray(ctx.dataType,n*c*h*w)
cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
alpha = 1.0
beta = 0.0
cudnnActivationForward(ctx.handle,1,alpha,ctx.srcTensorDesc,inputs,beta,ctx.dstTensorDesc,out)
return out
end

#TODO: still need to be change to fit the ADnode 
function activationBackward(ctx::CuDNNContext,inputs::CudaArray,outputs::CudaArray,diffDesc::cudnnTensorDescriptor_t,diff::CudaArray,n,c,h,w)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,n*c*w*h)
cudnnActivationBackward(ctx.handle,1,alpha,ctx.dstTensorDesc,ouputs,diffDesc,diff,ctx.srcTensorDesc,inputs,beta,diffDesc,diff,ctx.srcTensorDesc,out)
return out
end

