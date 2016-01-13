#TODO CPU Version
function FCUrectlin(inputs::Array,t::TensorDesc)



end
#TODO this is the CPU version
function DCUrectlin()


end


#TODO missing handler
function FCUrectlin(handler::cudnnHandle_t,inputs::CudaArray,t::TensorDesc)
context = CuDNNContext(handler,0,eltype(inputs))
rectlinNode = linearRectifierSetup(context,t.bancth,t.channel,t.height,t.width)
out = forward(context,rectlinNode,inputs)
free(context)
return out
end

function DCUrectlin()


end

CUrectlin(i::ADnode,t::TensorDesc,)=ADnode(FCUrectlin,[i t])