function gradcheckGPU(netGPU;showgrad=false)
# It's a bit painful to write a gradchecker for the GPU.
# We'll therefore check whether the GPU gradient matches the CPU gradient (which we can independently check is OK).

println("GRADCHECK GPU")

#netCPU=deepcopy(netGPU)
#for i in netCPU.node
#    netCPU.value[i]=to_host(netGPU.value[i])
#    netCPU.gradient[i]=to_host(netGPU.gradient[i])
#end
netCPU=convert(netGPU,"CPU")

netCPU=compile(netCPU;gpu=false)

ADforward!(netCPU); ADbackward!(netCPU); #println("evaluated CPU")
ADforward!(netGPU); ADbackward!(netGPU); #println("evaluated GPU")

    tol=1e-5
    for n in netGPU.validnodes # loop over the nodes in the network
        println("------------------------------------------------------------------------------")
        println("node $n:")
        
        D=length(netCPU.value[n])
        diff=sum(abs(netCPU.value[n]-to_host(netGPU.value[n])))/D
        reldiff=sum(abs((netCPU.value[n]-to_host(netGPU.value[n]))./(realmin()+netCPU.value[n])))/D
        println("absolute difference between CPU and GPU value=$diff")
        println("relative difference between CUP and GPU value=$reldiff")
        
        if diff>tol
            if showgrad
                println("CPU value:"); println(netCPU.value[n])
                println("GPU value:"); println(netGPU.value[n])
            end
            print_with_color(:red,"failed: mismatch more than $tol\n")
        else
            print_with_color(:green,"passed\n")
        end
        
        if any(node[n].takederivative)
            diff=sum(abs(netCPU.gradient[n]-to_host(netGPU.gradient[n])))/D
            reldiff=sum(abs((netCPU.gradient[n]-to_host(netGPU.gradient[n]))./(realmin()+netCPU.gradient[n])))/D
            println("absolute difference between CPU and GPU gradient=$diff")
            println("relative difference between CUP and GPU gradient=$reldiff")
            
            if diff>tol
                if showgrad
                    println("CPU gradient:"); println(netCPU.gradient[n])
                    println("GPU gradient:"); println(netGPU.gradient[n])
                end
                print_with_color(:red,"failed: mismatch more than $tol\n")
            else
                print_with_color(:green,"passed\n")
            end
            
        end
    end
end
