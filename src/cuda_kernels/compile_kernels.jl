kernels=["vcopyshift",
         "vcopyfrom",
         "vcopyfrom_update",
         "vsign",
         "gfill",
         "tx1mx",
         "Dstanh",
         "DmeanSquareLoss",
         "stanh",
         "binaryentropy",
         "binaryentropyXsigmoidY",
         "DXbinaryentropy",
         "DYbinaryentropy",
         "DXbinaryentropyXsigmoidY",
         "DYbinaryentropyXsigmoidY",
         "sigmoid",
         "vsquare",
         "exp",
         "log",
         "rectlin",
         "A_emult_Bg0",
         "alphaaxpy",
         "alphaax",
         "gaxpy",
         "gscale",
         "diag",
         "diagm",
         "gax",
         "ax",
         "vmultbang",
         "vdivbang",
         "vmultbangupdate",
         "vdivbangupdate",
         "vAoverBupdate",
         "vabs",
         "xsigny_update",
         "kinklin"
         ]

#cd("./cuda_kernels")
for k in kernels
    s=k*".cu"
    println(s)
#    run(`nvcc -arch sm_30 -use_fast_math -ptx $s`) # Jetson TK1
    run(`nvcc -use_fast_math -ptx $s`) # Titan GTX
end
#cd("../")
