type FilterDesc
inputs::Int
outputs::Int
kernelDim::Int
end

#TODO this is the CPU version 
function FConvolution(inputs::Array,filters::Array)



end
#TODO this is the CPU version
function DConvolution()


end


#TODO missing handler
function FConvolution(inputs::CudaArray,t::TensorDesc,filters::CudaArray,f::FilterDesc)
global handler
context = CuDNNContext(handler,0,eltype(inputs))
out = forward(context,inputs,filters,t.n,t.c,t.w,t.h,f.inputs,f.outputs,f.kernelDim)
return out
end

function DConvolution(handler::cudnnHandle_t,inputs::CudaArray,filters::CudaArray)
println("DConvolution called")
end

Derivative[FConvolution] = DConvolution
Inplace[FConvolution]   = FConvolution
export FConvolution
Convolution(inputs::ADnode,t::TensorDesc,filters::ADnode,f::FilterDesc)=ADnode(FConvolution,[inputs t filters f])
export Convolution

