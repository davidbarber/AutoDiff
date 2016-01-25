include("CUDA/CuDNN.jl")
include("CuDNNContext.jl")


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


