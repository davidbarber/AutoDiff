function ADforward!(net;returnf=false,debug=false)
#=   calculate the value and auxiliary values of a net function
=# 
    # (c) David Barber, University College London 2015
    #=
    if AllocateMemory & net.gpu
        TransformToGPU=true
        net=convert(net,"CPU")
    else
        TransformToGPU=false
    end
    =#

    if debug; println("Get value:"); end

    if isempty(net.auxvalue)
        net.auxvalue=similar(net.value)
    end

    # forward pass:
    for node in net.forwardNodes
        if debug
            println("node $(node.index): $(node.f)($(node.parents))");
        end
        
            if debug
                println("in place")
                if(isa(node.malloc,Bool))
                @time net.value[node] = node.f_inplace(net.handle,net.value[node],net.auxvalue[node],net.value[node.parents]...)
                else
                net.value[node] = node.f_inplace(net.handle,node.malloc,net.value[node],net.auxvalue[node],net.value[node.parents]...)
                end
            else
                if(isa(node.malloc,Bool))
                net.value[node] = node.f_inplace(net.handle,net.value[node],net.auxvalue[node],net.value[node.parents]...)
                else
                net.value[node] = node.f_inplace(net.handle,node.malloc,net.value[node],net.auxvalue[node],net.value[node.parents]...)
                end
            end
    end















    #=
    if TransformToGPU
        net=convert(net,"GPU")
    end

    if returnf
        if net.gpu
            return to_host(net.value[net.FunctionNode])[1]
        else
            return net.value[net.FunctionNode][1]
        end
#        @gpu if net.gru
#        @cpu return net.value[net.FunctionNode][1]
    end

    if debug; println("done"); end
    =#
end

