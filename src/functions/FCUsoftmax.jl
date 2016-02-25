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
alpha = 1.0
beta = 0.0
cudnnSoftmaxBackward(tensor.handle,1,1,alpha,tensor.tensorDesc,tensor.data.ptr,tensor.tensorDesc,grad_c.ptr,beta,tensor.tensorDesc,grad_n.ptr)
end


Derivative[FCUsoftmax] = DCUsoftmax
Inplace[FCUsoftmax]   = FCUsoftmax
CUsoftmax(i::ADTensor)=ADFunction(FCUsoftmax,i)
export CUsoftmax
