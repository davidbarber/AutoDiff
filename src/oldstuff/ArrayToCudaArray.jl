function ArrayToCudaArray!(net,inds) 
    for n in inds
        net.value[n]=CudaArray(net.value[n])
    end
end
