function usegpu(gpu::Bool)
global GPU=gpu
f=open("initfile.jl","w")
    if gpu==true
        write(f,"GPU=true\n")
    else
        write(f,"GPU=false\n")
    end
close(f)
end
