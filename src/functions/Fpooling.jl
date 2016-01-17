#TODO CPU version 
function FPooling(inputs::Array,t::TensorDesc)



end
#TODO this is the CPU version
function DPooling()


end


function FPooling(handler::cudnnHandle_t,inputs::CudaArray)
context = CuDNNContext(handler,0,eltype(inputs))
poolingNode = poolingSetup(context,t.bancth,t.channel,t.height,t.width)
out = forward(context,poolingNode,inputs)
free(context)
return out
end

function DPooling()


end

Pooling(i::ADnode)=ADnode(FPooling,[i])