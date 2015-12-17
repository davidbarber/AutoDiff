function ADforward!(net;returnf=false,exclude=Array(Int,0),debug=false,AllocateMemory=false)
#=   calculate the value and auxiliary values of a net function
    exclude is a collection of nodes to leave out of the forward pass calculation -- any descendents of these nodes are also automatically excluded
=#
    # (c) David Barber, University College London 2015

    if debug; println("Get value:"); end

    if isempty(net.auxvalue)
        net.auxvalue=similar(net.value)
    end

    # forward pass:
    for i in net.ForwardPassList
        thisnode=net.node[i]
        if debug
            println("node $i: $(thisnode.f)($(thisnode.parents))");
        end
        if AllocateMemory
            if debug
                @time net.value[i],net.auxvalue[i]=thisnode.f(net.value[thisnode.parents]...) ## not in place
            else
                net.value[i],net.auxvalue[i]=thisnode.f(net.value[thisnode.parents]...) ## not in place
            end
        else
            if debug
                println("in place")
                @time thisnode.f_inplace(net.value[i],net.auxvalue[i],net.value[thisnode.parents]...) ## in place
            else
                thisnode.f_inplace(net.value[i],net.auxvalue[i],net.value[thisnode.parents]...) ## in place
            end
        end
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
end

