function gradcheckCPU(net;showgrad=false,returngrad=false)
    #ONLY WORKS ON THE CPU AT THE MOMENT
    # (c) David Barber, University College London 2015
    println("GRADCHECK CPU")
    value=net.value
    node=net.node
    net2=deepcopy(net) ## CAREFUL -- DEEPCOPY IS BUGGY FOR CUDAARRAY

    ADforward!(net)
    ADbackward!(net)
    g=net.gradient

    epsilon=1e-7;
    tol=1e-6
    gemp=deepcopy(g)
    #for n=1:length(net.node) # loop over the nodes in the network
    for n in net.validnodes # loop over the nodes in the network
        if node[n].returnderivative
            println("------------------------------------------------------------------------------")
            println("node $n:")
            for linearidx in 1:prod(size(value[n]))
                net2.value[n][linearidx]=value[n][linearidx]+epsilon
                fplus=ADforward!(net2,returnf=true)
                net2.value[n][linearidx]=value[n][linearidx]-epsilon
                fminus=ADforward!(net2,returnf=true)
                net2.value[n][linearidx]=value[n][linearidx]
                gemp[n][linearidx]=0.5*(fplus-fminus)/epsilon
            end
            diff=mean(abs(g[n]-gemp[n]))
            reldiff=mean(abs((g[n]-gemp[n])./(realmin()+g[n])))
            println("absolute difference between analytic and empirical gradient=$diff")
            println("relative difference between analytic and empirical gradient=$reldiff")

                #if showgrad
                #println("analytic gradient:"); println(g[n])
                #println("empirical gradient:"); println(gemp[n])
                #end

            if diff>tol
                if showgrad
                println("analytic gradient:"); println(g[n])
                println("empirical gradient:"); println(gemp[n])
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
