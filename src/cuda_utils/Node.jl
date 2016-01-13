
include("CUDA/CuDNN.jl")


include("CuDNNContext.jl")

abstract Node

type ConvNode <: Node
convDesc::cudnnConvolutionDescriptor_t
filterDesc::cudnnFilterDescriptor_t
biasDesc::cudnnTensorDescriptor_t
algo::Int
size::Int
end

#TODO might need to change to ADNode in future ?
type TensorDesc
height::Int
width::Int
channel::Int
bancth::Int
end
#TODO might need to change to ADNode in future ?
type FilterDesc
inputs::Int
outputs::Int
kernelDims::Int
end

function convolutionSetup(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int,inputs::Int,outputs::Int,kDim::Int)
dType = cudnnDataTypeCheck(ctx.dataType)
cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
const dims = 4
tensorOutputDim = [n,c,h,w]
filterDim = [outputs,inputs,kDim,kDim]
filterDesc = cudnnCreateFilterDescriptor()
cudnnSetFilterNdDescriptor(filterDesc,net.dataType,dims,filterDim)
const convDims = 2;
padA= [0,0];
filterStrideA=[1,1];
upscaleA = [1,1];
CUDNN_CROSS_CORRELATION = 1;
cudnnSetConvolutionNdDescriptor(node.convDesc,convDims,padA,filterStrideA,upscaleA,CUDNN_CROSS_CORRELATION,net.dataType)
tensorOutputDimA = cudnnGetConvolutionNdForwardOutputDim(node.convDesc,net.srcTensorDesc,node.filterDesc,dims,tensorOutputDim)
n=tensorOutputDimA(1)
c=tensorOutputDimA(2)
h=tensorOutputDimA(3)
w=tensorOutputDimA(4)
convDesc = cudnnCreateConvolutionDescriptor()
cudnnSetTensorNdDescriptor(net.dstTensorDesc,net.tensorFormat,net.dataType,n,c,h,w)
algo = cudnnGetConvolutionForwardAlgorithm(ctx.handle,ctx.srcTensorDesc,ctx.filterDesc,convDesc,ctx.dstTensorDesc,1,0)

biasDesc = cudnnCreateTensorDescriptor()
cudnnSetTensorNdDescriptor(biasDesc,ctx.tensorFormat,ctx.dataType,1,c,1,1)
return ConvNode(convDesc,filterDesc,biasDesc,algo,n*h*c*w)
end

#TODO add bias
function forward(ctx::CuDNNContext,node::ConvNode,inputs::CudaArray,filters::CudaArray)
out = CudaArray(ctx.dataType,node.size)
alpha =1.0
beta = 0.0
sizeInByte = cudnnGetConvolutionForwardWorkspaceSize(ctx.handle,ctx.srcTensorDesc,node.filterDesc,node.convDesc,ctx.dstTensorDesc,algo)
workspace = nothing
if(sizeInByte ==0)
workspace = CudaPtr()
else
workspace =malloc(sizeInByte)
end
cudnnConvolutionForward(ctx.handle,alpha,ctx.srcTensorDesc,ctx.srcData,node.filterDesc,filters,node.convDesc,node.algo,workspace,sizeInByte,beta,node.destDesc,out)
return out
end




#TODO:wrap backward
function backward()

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
CuDNN.cudnnSetPoolingNdDescriptor(poolingDesc,0,dims,windowDim,padding,stride)

CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)

const tensorDims = 4
tensorOutPutDims =[n,c,h,w]
outputDim=CuDNN.cudnnGetPoolingNdForwardOutputDim(poolingDesc,ctx.srcTensorDesc,tensorDims,tensorOuputDims)
n=outputDim[1]
c=outputDim[2]
h=outputDim[3]
w=outputDim[4]

CuDNN.cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
return PoolingNode(poolingDesc,n*c*h*w)
end

function forward(ctx::CuDNNContext,node::PoolingNode,inputs)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,node.size)
CuDNN.cudnnPoolingForward(ctx.handle,node.poolingDesc,alpha,ctx.srcTensorDesc,inputs,beta,ctx.destDesc,out)
return out
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

function forward(ctx::CuDNNContext,node::LinearRectifierNode,inputs::CudaPtr)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,node.size)
CuDNN.cudnnLRNCrossChanelForward(ctx.cudnnHandle,node.normDesc,0,alpha,ctx.srcTensorDesc,inputs,beta,ctx.destDesc,out)
return out
end

#TODO: wrap backward
function backward()

end

function free()

end


#softmax
function softmaxForward(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int,inputs::CudaPtr)
dType = cudnnDataTypeCheck(ctx.dataType)
out = CudaArray(ctx.dataType,node.size)
CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
CuDNN.cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
alpha = 1.0
beta = 0.0
CuDNN.cudnnSoftmaxForward(ctx.handle,1,1,alpha,ctx.srcTensorDesc,inputs,beta,ctx.dstTensorDesc,out)
return out
end

#TODO: wrapping this function
function softmaxBackward()

end

#activation
function activationForward(ctx::CuDNNContext,n::Int,c::Int,h::Int,w::Int,inputs::CudaPtr)
dType = cudnnDataTypeCheck(ctx.dataType)
out = CudaArray(ctx.dataType,node.size)
CuDNN.cudnnSetTensorNdDescriptor(ctx.srcTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
CuDNN.cudnnSetTensorNdDescriptor(ctx.dstTensorDesc,ctx.tensorFormat,dType,n,c,h,w)
alpha = 1.0
beta = 0.0
CuDNN.cudnnActivationForward(ctx.handle,1,alpha,ctx.srcTensorDesc,inputs,beta,ctx.dstTensorDesc,out)
return out
end

function activationBackward()

end


#function resize(ctx::CuDNNContext,size::Int)
#if(data.p != nothing)
#CUDA.free(ctx.dstData)
#end
#CUDA.cualloc(ctx.dataType,size)
#end
