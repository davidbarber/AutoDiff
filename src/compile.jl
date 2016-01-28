# (c) David Barber, University College London 2015
#=From the README the main perporse of compile is to allocate memory
  For GPU or CPU operation
=#

function compile(net;backend="CPU",debug=false)
    

    while(isa(net.forwardNodes[1],ADconst))
    constant = shift!(net.forwardNodes)
    net.value[constant.index] = constant.value
    end
    
    if backend == "CPU"
        
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

    else
        throw("backend type must be GPU or CPU")
    end
    return net

end

