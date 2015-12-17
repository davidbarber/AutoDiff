
if ~isdefined(:GPU)
    GPU=false;
    CPU=true
end
macro gpu(ex); if GPU; esc(ex); end; end
macro cpu(ex); if ~GPU; esc(ex); end; end
