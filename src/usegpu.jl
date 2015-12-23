function usegpu(gpu::Bool)
global GPU=gpu
global CPU=~gpu
    
    f=open(ADDIR*"/src/initfile.jl","w")
    if gpu==true
        write(f,"GPU=true\n")
    else
        write(f,"GPU=false\n")
    end
close(f)
end
