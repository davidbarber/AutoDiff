function ADbackward!(net;debug=false,AccumulateGradient=false)

    #=   calculate the gradient of a net function
    (c) David Barber, University College London 2015
    =#

    if debug; println("Get gradient:"); end

    fill!(net.gradient[net.FunctionNode],1.0)
    for n in net.ancestors
        if net.node[n].takederivative
            if !AccumulateGradient
                fill!(net.gradient[n],0.0)
            end
            for c in net.relevantchildren[n]
                if debug
                    println("-----------------")
                    println("node $n, child $c: $(net.node[c].df)($(net.node[c].parents))")
                     for parIDX in net.parentIDX[c,n] # deals with the case that x-->f<--x (which happens with e.g. x+x)
                        println("parent index [$parIDX]")
                        # This is why gradients must always add up whatever is currently there (ie not replace).
                        @time net.node[c].df(parIDX,net.value[c],net.auxvalue[c],net.gradient[c],net.gradient[n],net.value[net.node[c].parents]...)
                    end
                else
                    for parIDX in net.parentIDX[c,n] # deals with the case that x-->f<--x (which happens with e.g. x+x)
                        net.node[c].df(parIDX,net.value[c],net.auxvalue[c],net.gradient[c],net.gradient[n],net.value[net.node[c].parents]...)
                    end
                end
            end
        end
    end

    if debug; println("done backward pass");  end

end

