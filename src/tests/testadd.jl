using CUDArt, PyPlot

CUDArt.init([0])

md=CuModule("vadd.ptx",false)
kernel=CuFunction(md,"vadd")

function vadd(A::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(kernel,nblocks,1024,(length(A),A,B,C))
end

N=2000
A=CudaArray(rand(N,N))
B=CudaArray(rand(N,N))
C=CudaArray(zeros(N,N))

M=5000
tm=zeros(M)
hA=to_host(A);
hB=to_host(B);
hC=to_host(C);

GPU=true
tic()
for i in 1:M
    if GPU vadd(A,B,C)
    else
        hC=hA+hB
    end
end
if GPU CUDArt.device_synchronize() end
#to_host(C)
avtime=toc()/M;
println("average time per vadd = $avtime")

