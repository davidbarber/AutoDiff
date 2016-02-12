global PROC
#global GPU
#global CPU

#if ~isdefined(:GPU)
#    GPU=false;
#    CPU=true
#end

if ~isdefined(:PROC)
    PROC="CPU"
end


#macro gpu(ex); if PROC=="GPU"; esc(ex); end; end
#macro cpu(ex); if ~GPU; esc(ex); end; end

macro gpu(ex)
    global PROC
    if PROC=="GPU" || PROC=="GPU32"
         esc(ex)
    end
end

macro cpu(ex)
    global PROC
    if PROC=="CPU"
         esc(ex)
    end
end

#export @gpuex, @cpuex
