function ADbackward!(net;debug=false,AccumulateGradient=false)

    #=   calculate the gradient of a net function
    (c) David Barber, University College London 2015
    =#


    if debug; println("Get gradient:"); end

    for node in net.backwardNodes
             #############IMPORTANT#############
            # all the grediants are accumulated #
            #          CHECK WITH DAVID         #
            
        for c in node.children
            derivativeIDX = first(findin(node.parents,c))

                if debug
                    println("-----------------")
                    println("node $(node.index), child $c: $(node.df)($(node.parents))")
                        # This is why gradients must always add up whatever is currently there (ie not replace)
                       if(isa(node.malloc,Bool))
                        @time net.gradient[c]=node.df(net.handle,derivativeIDX,net.value[node],net.auxvalue[node],net.gradient[node],net.gradient[c],net.value[node.parents]...)
                        else
                         @time net.gradient[c]=node.df(net.handle,derivativeIDX,net.value[node],net.auxvalue[node],net.gradient[node],net.gradient[c],net.value[node.parents]...)
                        end


                    else
                    if(isa(node.malloc,Bool))
                         net.gradient[c]=node.df(net.handle,derivativeIDX,net.value[node],net.auxvalue[node],net.gradient[node],net.gradient[c],net.value[node.parents]...)
                        else
                         net.gradient[c]=node.df(net.handle,node.malloc,derivativeIDX,net.value[node],net.auxvalue[node],net.gradient[node],net.gradient[c],net.value[node.parents]...)
                        end


                 # deals with the case that x-->f<--x (which happens with e.g. x+x)
                end
        end
    end
    

    if debug
    println("done backward pass")
    end

end

