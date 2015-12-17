function gradcheck(net;showgrad=false,returngrad=false)
    #ONLY WORKS ON THE CPU AT THE MOMENT
    # (c) David Barber, University College London 2015
    println("gradcheck...")
    value=net.value
    node=net.node
    net2=deepcopy(net) ## CAREFUL -- DEEPCOPY IS BUGGY FOR CUDAARRAY

    ADforward!(net)
    ADbackward!(net)
    g=net.gradient

    epsilon=1e-7;
    tol=1e-6
    gemp=deepcopy(g)
    for par=1:length(net.node) # loop over the nodes in the network
        if node[par].returnderivative
            println("------------------------------------------------------------------------------")
            println("node $par:")
            for linearidx in 1:prod(size(value[par]))
                net2.value[par][linearidx]=value[par][linearidx]+epsilon
                fplus=ADforward!(net2,returnf=true)
                net2.value[par][linearidx]=value[par][linearidx]-epsilon
                fminus=ADforward!(net2,returnf=true)
                net2.value[par][linearidx]=value[par][linearidx]
                gemp[par][linearidx]=0.5*(fplus-fminus)/epsilon
            end
            diff=mean(abs(g[par]-gemp[par]))
            reldiff=mean(abs((g[par]-gemp[par])./(realmin()+g[par])))
            println("absolute difference between analytic and empirical gradient=$diff")
            println("relative difference between analytic and empirical gradient=$reldiff")

                #if showgrad
                #println("analytic gradient:"); println(g[par])
                #println("empirical gradient:"); println(gemp[par])
                #end

            if diff>tol
                if showgrad
                println("analytic gradient:"); println(g[par])
                println("empirical gradient:"); println(gemp[par])
                end
                print_with_color(:red,"failed: analytic and empiricial gradient mismatch more than $tol\n")
            else
                print_with_color(:green,"passed\n")
            end
        end
    end
    if returngrad
        return gemp
    end
end
