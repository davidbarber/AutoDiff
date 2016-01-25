#TODO CPU Version
function FCUsoftmax(inputs::Array)

println("CPU version of CuDNN softmax under Development")
return inputs,inputs
end
#TODO this is the CPU version
function DCUsoftmax()
println("CPU version of CuDNN softmax under Development")
end



function FCUsoftmax(value::CudaArray,au::CudaArray,tensor::CudaTensor)
dstTensorDesc = cudnnCreateTensorDescriptor()
dimension = tensor.dims
cudnnSetTensor4dDescriptor(dstTensorDesc,tensor.dataType,dimension[1],dimension[2],dimension[3],dimension[4])
alpha = 1.0
beta = 0.0
cudnnSoftmaxForward(tensor.handle,1,1,alpha,tensor.tensorDesc,tensor.data.ptr,beta,dstTensorDesc,value.ptr)
end
export FCUsoftmax

function DCUsoftmax(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaTensor)
println(typeof(derivativeIDX))
println(typeof(f_c))
println(typeof(faux_c))
println(typeof(grad_c))
println(typeof(grad_n))
end

Derivative[FCUsoftmax] = DCUsoftmax
Inplace[FCUsoftmax]   = FCUsoftmax
CUsoftmax(i::ADTensor)=ADnode(FCUsoftmax,[i])
export CUsoftmax


#TODO: still need to be change to fit ADnode
function softmaxBackward(inputs::CudaArray,ouputs::CudaArray,diffDesc::cudnnTensorDescriptor_t,diff::CudaArray,n,w,h,c)
alpha = 1.0
beta = 0.0
out = CudaArray(ctx.dataType,n*w*h*c)
cudnnSoftmaxBackward(ctx.handle,1,1,alpha,ctx.dstTensorDesc,outputs,diffDesc,diff,beta,ctx.srcTensorDesc,out)
end