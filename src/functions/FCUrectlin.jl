#TODO CPU Version
function FCrossChannel!(inputs::Array)

println("CPU rectLin in development")
return inputs,inputs
end
#TODO this is the CPU version
function DCUrectlin()

end


#TODO missing handler
function FCrossChannel!(handle,value::CudaArray,auxvalue,X::CudaArray)
#creation 

free(value)
(n,c,h,w) = size(X)
dtype = eltype(X)
dataType = cudnnDataTypeCheck(dtype)
srcDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(srcDataDesc,dataType,n,c,h,w)
value = CudaArray(dtype,(n,c,h,w))

normDesc = cudnnCreateLRNDescriptor()
N = 5
alpha = 0.0001
beta = 0.75
K = 1.0
cudnnSetLRNDescriptor(normDesc,N,alpha,beta,K)
dstDataDesc = cudnnCreateTensorDescriptor()
cudnnSetTensor4dDescriptor(dstDataDesc,dataType,n,c,h,w)
cudnnLRNCrossChanelForward(handle,normDesc,0,alpha,srcDataDesc,X.ptr,beta,dstDataDesc,value.ptr)
return value
end

function DCrossChannel(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,X::CudaArray)
# grad_n child
# grad_c current 

end

Derivative[FCrossChannel!] = DCrossChannel
Inplace[FCrossChannel!]   = FCrossChannel!
CrossChannel(i::ADnode)=ADFunction(FCrossChannel!,i)
export CrossChannel



#TODO: still need to be change to fit ADNode
#=
function backward(node::LinearRectifierNode,inputs::CudaArray,outputs::CudaArray,diffDesc::cudnnTensorDescriptor_t,diff::CudaArray)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,node.size)
cudnnLRNCrossChanelBackward(ctx.handle,node.normDesc,0,alpha,ctx.dstTensorDesc,outputs,diffDesc,diff,ctx.srcTensorDesc,inputs,beta,ctx.srcTensorDesc,out)
return out
end
=#
