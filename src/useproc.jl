function useproc(proc)
    global PROC
    if (proc=="CPU") | (proc=="GPU")
        PROC=proc
    end
    include(ADDIR*"/src/gpumacros.jl")
    
    f=open(ADDIR*"/src/initfile.jl","w")
    if proc=="GPU"
        write(f,"global PROC; PROC=\"GPU\"\n")
    else
        write(f,"global PROC; PROC=\"CPU\"\n")
    end
    close(f)
end
