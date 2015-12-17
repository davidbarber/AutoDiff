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
         "vAoverBupdate"
         ]

ARCH="sm_30" # tested for Jetson TK1. Other architecture may need different arch setting.
FASTMATH="-use_fast_math"
for k in kernels
    s="./cuda_kernels/"*k*".cu"
    println(s)
    run(`nvcc -arch $ARCH $FASTMATH -ptx $s`)
end
