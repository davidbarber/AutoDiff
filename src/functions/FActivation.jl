#TODO CPU Version
function FActivation(inputs::Array,t::TensorDesc)



end
#TODO this is the CPU version
function DActivation()


end


#TODO missing handler
function FActivation(handler::cudnnHandle_t,inputs::CudaArray,t::TensorDesc)
context = CuDNNContext(handler,0,eltype(inputs))
out = activationForward(context,t.bancth,t.channel,t.height,t.width,inputs)

free(context)
return out
end

function DActivation()


end

CUsoftmax(i::ADnode,t::TensorDesc,)=ADnode(FCUsoftmax,[i t])