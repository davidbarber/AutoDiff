function useproc(proc)
    global PROC
    if (proc=="CPU") | (proc=="GPU") | (proc=="GPU32")
        PROC=proc
    end
    include(ADDIR*"/src/gpumacros.jl")
    
    f=open(ADDIR*"/src/initfile.jl","w")
    if (proc=="GPU") 
        write(f,"global PROC; PROC=\"GPU\"\n")
    end
    if (proc=="GPU32") 
        write(f,"global PROC; PROC=\"GPU32\"\n")
    end
    if (proc=="CPU") 
        write(f,"global PROC; PROC=\"CPU\"\n")
    end
    close(f)
end
