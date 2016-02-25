
function compile(net;debug=false,backend="CPU",eltype=Float64)
    # (c) David Barber, University College London 2015
# (c) David Barber, University College London 2015
#=From the README the main perporse of compile is to allocate memory
  For GPU or CPU operation
=#   
    # put constants into value 
    while(isa(net.forwardNodes[1],ADconst))
    constant = shift!(net.forwardNodes)
    net.value[constant.index] = constant.value
    end
    
    if backend == "CPU"
        for node in net.forwardNodes
        


        end
        # TODO: for loop below can be saved, as filter() ADforward() also iterates 
        # through all node
        ADforward!(net;debug=debug,AllocateMemory=true)
        #Allocate graidents
        iter = length(net.value)
        net.gradient[iter] = cArray(false,Float64,size(net.value[iter]))
            for i = 1:iter-1
            net.gradient[i]=cArray(false,Float64,size(net.value[i]))
            end

        elseif backend == "GPU"
        net.handle = cudnnCreate()
        # TODO: for loop below can be saved, as filter() ADforward() also iterates 
        # through all node
        ADforward!(net;debug=debug,AllocateMemory=true)

        iter = length(net.value)
        s = size(net.value[iter])
        net.gradient[iter] = cArray(true,ones(s))
        net.value[iter] = cArray(true,net.value[iter])

        for i=1:iter-1
        s = size(net.value[i])
        net.gradient[i] = cArray(true,zeros(s))
        net.value[i] = cArray(true,net.value[i])
        end
    end
#=
    if debug;  println("Done forward pass compilation:");  end

    # backward compilation:
    if debug;  println("Backward Pass compilation:");  end

    gradient=Array(Any,N)

    for i in net.validnodes
        gradient[i]=cArray(gpu,Float64,size(value[i]))
    end
    fill!(gradient[net.FunctionNode],1.0)

    Nend=net.FunctionNode
    if Nend!=N
        anc=ancestors(node,Nend)

    else
        throw("backend type must be GPU or CPU")
    end

    parentIDX=Dict()
    for c in net.validnodes
        for n in unique(node[c].parents)
            inds=findin(node[c].parents,n)
            parentIDX[c,n]=inds
        end
    end

    net.ancestors=anc
    net.relevantchildren=relevantchildren
    net.gradient=gradient
    net.parentIDX=parentIDX

    if debug
        println("Compiled into a DAG with $(length(node)) nodes")
    end
=#

    return net

end

