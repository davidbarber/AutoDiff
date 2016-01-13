
#TODO this is the CPU version 
function FConvolution(inputs::Array,filters::Array,t::TensorDesc,f::FilterDesc)



end
#TODO this is the CPU version
function DConvolution()


end


#TODO missing handler
function FConvolution(handler::cudnnHandle_t,inputs::CudaArray,filters::CudaArray,t::TensorDesc,f::FilterDesc)
context = CuDNNContext(handler,0,eltype(inputs))
convNode = convolutionSetup(context,t.bancth,t.channel,t.height,t.width,f.inputs,f.outputs,f.kernelDims)
out = forward(context,convNode,inputs,filters)
free(context)
return out
end

function DConvolution()


end

Convolution(i::ADnode,f::ADnode,t::TensorDesc,fd::FilterDesc)=ADnode(FConvolution,[i f t fd])

