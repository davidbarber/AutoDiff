#TODO CPU Version
function FCUrectlin(inputs::Array)

println("CPU rectLin in development")
return inputs,inputs
end
#TODO this is the CPU version
function DCUrectlin()

end


#TODO missing handler
function FCUrectlin(value::CudaArray,au::CudaArray,tensor::CudaTensor)
normDesc = cudnnCreateLRNDescriptor()
N = 5
alpha = 0.0001
beta = 0.75
K = 1.0
cudnnSetLRNDescriptor(normDesc,N,alpha,beta,K)
dstDesc = cudnnCreateTensorDescriptor()
dimension = tensor.dims
cudnnSetTensor4dDescriptor(dstDesc,tensor.dataType,dimension[1],dimension[2],dimension[3],dimension[4])
cudnnLRNCrossChanelForward(tensor.handle,normDesc,0,alpha,tensor.tensorDesc,tensor.data.ptr,beta,dstDesc,value.ptr)
end

function DCUrectlin()


end

Derivative[FCUrectlin] = DCUrectlin
Inplace[FCUrectlin]   = FCUrectlin
CUrectlin(i::ADTensor)=ADFunction(FCUrectlin,i)
export CUrectlin



#TODO: still need to be change to fit ADNode
function backward(node::LinearRectifierNode,inputs::CudaArray,outputs::CudaArray,diffDesc::cudnnTensorDescriptor_t,diff::CudaArray)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,node.size)
cudnnLRNCrossChanelBackward(ctx.handle,node.normDesc,0,alpha,ctx.dstTensorDesc,outputs,diffDesc,diff,ctx.srcTensorDesc,inputs,beta,ctx.srcTensorDesc,out)
return out
end
