
function compile(net,backend;debug=false,eltype=Float64)
    # (c) David Barber, University College London 2015
# (c) David Barber, University College London 2015
#From the README the main perporse of compile is to allocate memory
# For GPU or CPU operation  
# Memory allocation can still be save
# As if user choose to run on GPU
# it is not nessaccary to allcate memory on CPU first

    # put constants into net.value
    # allocate memory for ADVariable 
    # After this while loop nodes in forwardNodes should only be ADFunctionNode
    

    if backend =="CPU"
           # Allocate memory for inputs includes:
           #   i) Input variables
           #  ii) Parameters 
           # iii) Constant 
           # similar for GPU version
         while(!isa(net.forwardNodes[1],ADFunctionNode))
                node = shift!(net.forwardNodes)
    
            if(isa(node,ADconst))
                net.value[node.index] = node.value
                elseif(isa(node,ADVariable))
                net.gradient[node] = fill(0,size(net.value[node]))
            else
                continue
            end

         end

        for node in net.forwardNodes
            s =node.f(node.malloc,net.value[node.parents]...)
            net.value[node] = fill(0,s)
            net.auxvalue[node] = fill(0,s)
            net.gradient[node] = fill(0,s)
        end
        
    elseif backend =="GPU"
        net.handle = cudnnCreate() 
        # GPU size allocation still need to be optimised
        # GPU memory allocation can still be save by sharing pointer
        # and reduce memory transaction
        while(!isa(net.forwardNodes[1],ADFunctionNode))
                node = shift!(net.forwardNodes)
            if(isa(node,ADconst))
                net.value[node.index] = cArray(backend,node.value)
            elseif(isa(node,ADVariable))
                net.value[node] = cArray(backend,net.value[node])
                net.gradient[node] = cArray(backend,zeros(size(net.value[node])))
            else
                continue
            end

         end

        for node in net.forwardNodes
            s =node.f(node.malloc,net.value[node.parents]...)
            println(s)
            net.value[node] = cArray(backend,zeros(s))
            net.auxvalue[node] = cArray(backend,zeros(s))
            net.gradient[node] = cArray(backend,zeros(s))
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

