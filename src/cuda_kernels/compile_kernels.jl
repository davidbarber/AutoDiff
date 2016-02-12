kernels=filter(x->contains(x,".cu")&~contains(x,"~"),readdir("./"))

println("Compiling Kernels:")
for k in kernels
    println(k)
#    run(`nvcc -arch sm_30 -use_fast_math -ptx $k`) # Jetson TK1
    run(`/usr/local/cuda/bin/nvcc -c -use_fast_math -ptx $k`) # Titan GTX
end
