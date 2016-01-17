#TODO CPU Version
function FCUsoftmax(inputs::Array)



end
#TODO this is the CPU version
function DCUsoftmax()


end


#TODO missing handler
function FCUsoftmax(handler::cudnnHandle_t,inputs::CudaArray)
context = CuDNNContext(handler,0,eltype(inputs))
out = softmaxForward(context,t.bancth,t.channel,t.height,t.width,inputs)

free(context)
return out
end

function DCUsoftmax()


end

CUsoftmax(i::ADnode)=ADnode(FCUsoftmax,[i])