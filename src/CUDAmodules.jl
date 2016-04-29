DIR=pwd()*"/cuda_kernels/"

println("loading CUDA kernels from $DIR")

md=CuModule(DIR*"vcopyshift.ptx",false)
vcopyshift_kernel=CuFunction(md,"vcopyshift")
function copyinto!(out::CudaArray{Float64},A::CudaArray{Float64},location::Int)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(vcopyshift_kernel,nblocks,1024,(length(A),location-1,A,out))
end

md=CuModule(DIR*"vcopyshift_32.ptx",false)
vcopyshift_kernel_32=CuFunction(md,"vcopyshift_32")
function copyinto!(out::CudaArray{Float32},A::CudaArray{Float32},location::Int)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(vcopyshift_kernel_32,nblocks,1024,(length(A),location-1,A,out))
end
export copyinto!

md=CuModule(DIR*"vcopyfrom.ptx",false)
vcopyfrom_kernel=CuFunction(md,"vcopyfrom")
function copyfrom!(out::CudaArray{Float64},A::CudaArray{Float64},location::Int)
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vcopyfrom_kernel,nblocks,1024,(length(out),location-1,A,out))
end

md=CuModule(DIR*"vcopyfrom_32.ptx",false)
vcopyfrom_kernel_32=CuFunction(md,"vcopyfrom_32")
function copyfrom!(out::CudaArray{Float32},A::CudaArray{Float32},location::Int)
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vcopyfrom_kernel_32,nblocks,1024,(length(out),location-1,A,out))
end
export copyfrom!

md=CuModule(DIR*"vcopyfrom_update.ptx",false)
vcopyfrom_update_kernel=CuFunction(md,"vcopyfrom_update")
function copyfrom_update!(out::CudaArray{Float64},A::CudaArray{Float64},location::Int)
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vcopyfrom_update_kernel,nblocks,1024,(length(out),location-1,A,out))
end

md=CuModule(DIR*"vcopyfrom_update_32.ptx",false)
vcopyfrom_update_kernel_32=CuFunction(md,"vcopyfrom_update_32")
function copyfrom_update!(out::CudaArray{Float32},A::CudaArray{Float32},location::Int)
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vcopyfrom_update_kernel_32,nblocks,1024,(length(out),location-1,A,out))
end
export copyfrom_update!

md=CuModule(DIR*"vsign.ptx",false)
vsign_kernel=CuFunction(md,"vsign")
function vsign!(A::CudaArray{Float64},out::CudaArray{Float64})
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vsign_kernel,nblocks,1024,(length(out),A,out))
end

md=CuModule(DIR*"vsign_32.ptx",false)
vsign_kernel_32=CuFunction(md,"vsign_32")
function vsign!(A::CudaArray{Float32},out::CudaArray{Float32})
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vsign_kernel_32,nblocks,1024,(length(out),A,out))
end
export vsign!

md=CuModule(DIR*"xsigny_update.ptx",false)
xsigny_update_kernel=CuFunction(md,"xsigny_update")
function xsigny_update!(X::CudaArray{Float64},Y::CudaArray{Float64},out::CudaArray{Float64})
    nblocks=round(Int,ceil(length(out)/1024))
    launch(xsigny_update_kernel,nblocks,1024,(length(out),X,Y,out))
end

md=CuModule(DIR*"xsigny_update_32.ptx",false)
xsigny_update_kernel_32=CuFunction(md,"xsigny_update_32")
function xsigny_update!(X::CudaArray{Float32},Y::CudaArray{Float32},out::CudaArray{Float32})
    nblocks=round(Int,ceil(length(out)/1024))
    launch(xsigny_update_kernel_32,nblocks,1024,(length(out),X,Y,out))
end
export xsigny_update!


md=CuModule(DIR*"vabs.ptx",false)
vabs_kernel=CuFunction(md,"vabs")
function abs!(A::CudaArray{Float64},out::CudaArray{Float64})
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vabs_kernel,nblocks,1024,(length(out),A,out))
end
export abs!



md=CuModule(DIR*"gfill.ptx",false)
gfill_kernel=CuFunction(md,"gfill")
function gfill!(out::CudaArray{Float64},fillval::CudaArray{Float64})
    nblocks=round(Int,ceil(length(out)/1024))
    launch(gfill_kernel,nblocks,1024,(length(out),fillval,out))
end

md=CuModule(DIR*"gfill_32.ptx",false)
gfill_kernel_32=CuFunction(md,"gfill_32")
function gfill!(out::CudaArray{Float32},fillval::CudaArray{Float32})
    nblocks=round(Int,ceil(length(out)/1024))
    launch(gfill_kernel_32,nblocks,1024,(length(out),fillval,out))
end
export gfill!

md=CuModule(DIR*"tx1mx_32.ptx",false)
tx1mx_kernel_32=CuFunction(md,"tx1mx_32")
function tx1mx!(t::CudaArray{Float32},x::CudaArray{Float32},out::CudaArray{Float32})
    nblocks=round(Int,ceil(length(x)/1024))
    launch(tx1mx_kernel_32,nblocks,1024,(length(x),t,x,out))
end

md=CuModule(DIR*"tx1mx.ptx",false)
tx1mx_kernel=CuFunction(md,"tx1mx")
function tx1mx!(t::CudaArray{Float64},x::CudaArray{Float64},out::CudaArray{Float64})
    nblocks=round(Int,ceil(length(x)/1024))
    launch(tx1mx_kernel,nblocks,1024,(length(x),t,x,out))
end
export tx1mx!

md=CuModule(DIR*"Dstanh.ptx",false)
Dstanh_kernel=CuFunction(md,"Dstanh")
function Dstanh!(sf,grad_c::CudaArray{Float64},f_c::CudaArray{Float64},grad_n::CudaArray{Float64})
    nblocks=round(Int,ceil(length(f_c)/1024))
    launch(Dstanh_kernel,nblocks,1024,(length(f_c),sf,grad_c,f_c,grad_n))
end

md=CuModule(DIR*"Dstanh_32.ptx",false)
Dstanh_kernel_32=CuFunction(md,"Dstanh_32")
function Dstanh!(sf,grad_c::CudaArray{Float32},f_c::CudaArray{Float32},grad_n::CudaArray{Float32})
    nblocks=round(Int,ceil(length(f_c)/1024))
    launch(Dstanh_kernel_32,nblocks,1024,(length(f_c),sf,grad_c,f_c,grad_n))
end
export Dstanh!

md=CuModule(DIR*"DmeanSquareLoss.ptx",false)
DmeanSquareLoss_kernel=CuFunction(md,"DmeanSquareLoss")
function DmeanSquareLoss!(grad_c::CudaArray{Float64},x::CudaArray{Float64},y::CudaArray{Float64},grad_n::CudaArray{Float64})
    nblocks=round(Int,ceil(length(x)/1024))
    launch(DmeanSquareLoss_kernel,nblocks,1024,(length(x),2.0/length(x),grad_c,x,y,grad_n))
end

md=CuModule(DIR*"DmeanSquareLoss_32.ptx",false)
DmeanSquareLoss_kernel_32=CuFunction(md,"DmeanSquareLoss_32")
function DmeanSquareLoss!(grad_c::CudaArray{Float32},x::CudaArray{Float32},y::CudaArray{Float32},grad_n::CudaArray{Float32})
    nblocks=round(Int,ceil(length(x)/1024))
    launch(DmeanSquareLoss_kernel_32,nblocks,1024,(length(x),Float32(2.0/length(x)),grad_c,x,y,grad_n))
end
export DmeanSquareLoss!



md=CuModule(DIR*"stanh.ptx",false)
stanh_kernel=CuFunction(md,"stanh")
function stanh!(sf,A::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(stanh_kernel,nblocks,1024,(length(A),sf,A,B))
end


md=CuModule(DIR*"stanh_32.ptx",false)
stanh_kernel_32=CuFunction(md,"stanh_32")
function stanh!(sf,A::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(stanh_kernel_32,nblocks,1024,(length(A),sf,A,B))
end
export stanh!



md=CuModule(DIR*"binaryentropy.ptx",false)
binaryentropy_kernel=CuFunction(md,"binaryentropy")
function binaryentropy(X::CudaArray{Float64},Y::CudaArray{Float64})
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropy_kernel,nblocks,1024,(length(X),X,Y,tmp))
    out=mean(tmp)
    free(tmp)
    return out
end

md=CuModule(DIR*"binaryentropy_32.ptx",false)
binaryentropy_kernel_32=CuFunction(md,"binaryentropy_32")
function binaryentropy(X::CudaArray{Float32},Y::CudaArray{Float32})
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropy_kernel_32,nblocks,1024,(length(X),X,Y,tmp))
    out=mean(tmp)
    free(tmp)
    return out
end
export binaryentropy

md=CuModule(DIR*"binaryentropyXsigmoidY.ptx",false)
binaryentropyXsigmoidY_kernel=CuFunction(md,"binaryentropyXsigmoidY")
function binaryentropyXsigmoidY(X::CudaArray{Float64},Y::CudaArray{Float64})
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropyXsigmoidY_kernel,nblocks,1024,(length(X),X,Y,tmp))
    out=mean(tmp)
    free(tmp)
    return out
end

md=CuModule(DIR*"binaryentropyXsigmoidY_32.ptx",false)
binaryentropyXsigmoidY_kernel_32=CuFunction(md,"binaryentropyXsigmoidY_32")
function binaryentropyXsigmoidY(X::CudaArray{Float32},Y::CudaArray{Float32})
    tmp=CudaArray(Float32,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropyXsigmoidY_kernel_32,nblocks,1024,(length(X),X,Y,tmp))
    out=mean(tmp)
    free(tmp)
    return out
end
export binaryentropyXsigmoidY

function binaryentropyXsigmoidY!(X::CudaArray{Float64},Y::CudaArray{Float64},out)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropyXsigmoidY_kernel,nblocks,1024,(length(X),X,Y,tmp))
    copy!(out,mean(tmp))
    free(tmp)
end

function binaryentropyXsigmoidY!(X::CudaArray{Float32},Y::CudaArray{Float32},out::CudaArray{Float32})
    tmp=CudaArray(Float32,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropyXsigmoidY_kernel_32,nblocks,1024,(length(X),X,Y,tmp))
    copy!(out,mean(tmp))
    free(tmp)
end
export binaryentropyXsigmoidY!

md=CuModule(DIR*"DXbinaryentropy.ptx",false)
DXbinaryentropy_kernel=CuFunction(md,"DXbinaryentropy")
function DXbinaryentropy!(X::CudaArray{Float64},Y::CudaArray{Float64},T::CudaArray{Float64},Out::CudaArray{Float64})
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DXbinaryentropy_kernel,nblocks,1024,(length(X),X,Y,T,Out))
end

md=CuModule(DIR*"DXbinaryentropy_32.ptx",false)
DXbinaryentropy_kernel_32=CuFunction(md,"DXbinaryentropy_32")
function DXbinaryentropy!(X::CudaArray{Float32},Y::CudaArray{Float32},T::CudaArray{Float32},Out::CudaArray{Float32})
    tmp=CudaArray(Float32,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DXbinaryentropy_kernel_32,nblocks,1024,(length(X),X,Y,T,Out))
end
export DXbinaryentropy!

md=CuModule(DIR*"DYbinaryentropy.ptx",false)
DYbinaryentropy_kernel=CuFunction(md,"DYbinaryentropy")
function DYbinaryentropy!(X::CudaArray{Float64},Y::CudaArray{Float64},T::CudaArray{Float64},Out::CudaArray{Float64})
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DYbinaryentropy_kernel,nblocks,1024,(length(X),X,Y,T,Out))
end

md=CuModule(DIR*"DYbinaryentropy_32.ptx",false)
DYbinaryentropy_kernel_32=CuFunction(md,"DYbinaryentropy_32")
function DYbinaryentropy!(X::CudaArray{Float32},Y::CudaArray{Float32},T::CudaArray{Float32},Out::CudaArray{Float32})
    tmp=CudaArray(Float32,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DYbinaryentropy_kernel_32,nblocks,1024,(length(X),X,Y,T,Out))
end
export DYbinaryentropy!


md=CuModule(DIR*"DXbinaryentropyXsigmoidY.ptx",false)
DXbinaryentropyXsigmoidY_kernel=CuFunction(md,"DXbinaryentropyXsigmoidY")
function DXbinaryentropyXsigmoidY!(X::CudaArray{Float64},Y::CudaArray{Float64},T::CudaArray{Float64},Out::CudaArray{Float64})
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DXbinaryentropyXsigmoidY_kernel,nblocks,1024,(length(X),X,Y,T,Out))
end

md=CuModule(DIR*"DXbinaryentropyXsigmoidY_32.ptx",false)
DXbinaryentropyXsigmoidY_kernel32=CuFunction(md,"DXbinaryentropyXsigmoidY_32")
function DXbinaryentropyXsigmoidY!(X::CudaArray{Float32},Y::CudaArray{Float32},T::CudaArray{Float32},Out::CudaArray{Float32})
    tmp=CudaArray(Float32,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DXbinaryentropyXsigmoidY_kernel32,nblocks,1024,(length(X),X,Y,T,Out))
end
export DXbinaryentropyXsigmoidY!


md=CuModule(DIR*"DYbinaryentropyXsigmoidY.ptx",false)
DYbinaryentropyXsigmoidY_kernel=CuFunction(md,"DYbinaryentropyXsigmoidY")
function DYbinaryentropyXsigmoidY!(X::CudaArray,Y::CudaArray,T::CudaArray,Out::CudaArray)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DYbinaryentropyXsigmoidY_kernel,nblocks,1024,(length(X),X,Y,T,Out))
end


md=CuModule(DIR*"DYbinaryentropyXsigmoidY_32.ptx",false)
DYbinaryentropyXsigmoidY_kernel32=CuFunction(md,"DYbinaryentropyXsigmoidY_32")
function DYbinaryentropyXsigmoidY!(X::CudaArray{Float32},Y::CudaArray{Float32},T::CudaArray{Float32},Out::CudaArray{Float32})
    tmp=CudaArray(Float32,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DYbinaryentropyXsigmoidY_kernel32,nblocks,1024,(length(X),X,Y,T,Out))
end
export DYbinaryentropyXsigmoidY!


md=CuModule(DIR*"sigmoid.ptx",false)
sigmoid_kernel=CuFunction(md,"sigmoid")
function sigmoid(A::CudaArray{Float64})
    out=CudaArray(Float64,size(A))
    nblocks=round(Int,ceil(length(A)/1024))
    launch(sigmoid_kernel,nblocks,1024,(length(A),A,out))
    return out
end

md=CuModule(DIR*"sigmoid32.ptx",false)
sigmoid_kernel_32=CuFunction(md,"sigmoid32")
function sigmoid(A::CudaArray{Float32})
    out=CudaArray(Float32,size(A))
    nblocks=round(Int,ceil(length(A)/1024))
    launch(sigmoid_kernel_32,nblocks,1024,(length(A),A,out))
    return out
end
export sigmoid

function sigmoid!(A::CudaArray{Float64},out::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(sigmoid_kernel,nblocks,1024,(length(A),A,out))
end

function sigmoid!(A::CudaArray{Float32},out::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(sigmoid_kernel_32,nblocks,1024,(length(A),A,out))
end
export sigmoid!


md=CuModule(DIR*"vsquare.ptx",false)
vsquarekernel=CuFunction(md,"vsquare")
function vsquare(A::CudaArray{Float64})
    out=zeros(Float64,A)
    launch(vsquarekernel,size(A,1),size(A,2),(A,out))
    return out
end

md=CuModule(DIR*"vsquare_32.ptx",false)
vsquarekernel_32=CuFunction(md,"vsquare_32")
function vsquare(A::CudaArray{Float32})
    out=zeros(Float32,A)
    launch(vsquarekernel_32,size(A,1),size(A,2),(A,out))
    return out
end


#function vsquareout!(A::CudaArray{Float64},Out::CudaArray{Float64})
#    launch(vsquarekernel,size(A,1),size(A,2),(A,Out))
#end

#function vsquareout!(A::CudaArray{Float32},Out::CudaArray{Float32})
#    launch(vsquarekernel_32,size(A,1),size(A,2),(A,Out))
#end


import Base.exp
md=CuModule(DIR*"exp.ptx",false)
kernel_exp=CuFunction(md,"expkernel")
function exp!(A::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_exp,nblocks,1024,(length(A),A,B))
end

md=CuModule(DIR*"exp_32.ptx",false)
kernel_exp_32=CuFunction(md,"expkernel_32")
function exp!(A::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_exp_32,nblocks,1024,(length(A),A,B))
end


function exp(A::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    B=CudaArray(Float64,size(A))
    launch(kernel_exp,nblocks,1024,(length(A),A,B))
    return B # MEMORY LEAK -- HOW TO REMOVE TEMPORARY B?
end

function exp(A::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    B=CudaArray(Float32,size(A))
    launch(kernel_exp_32,nblocks,1024,(length(A),A,B))
    return B # MEMORY LEAK -- HOW TO REMOVE TEMPORARY B?
end

export exp,exp!

import Base.log
md=CuModule(DIR*"log.ptx",false)
kernel_log=CuFunction(md,"logkernel")
function log!(A::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_log,nblocks,1024,(length(A),A,B))
end

md=CuModule(DIR*"log_32.ptx",false)
kernel_log_32=CuFunction(md,"logkernel_32")
function log!(A::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_log_32,nblocks,1024,(length(A),A,B))
end

function log(A::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    B=CudaArray(Float64,size(A))
    launch(kernel_log,nblocks,1024,(length(A),A,B))
    return B # MEMORY LEAK -- HOW TO REMOVE TEMPORARY B?
end

function log(A::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    B=CudaArray(Float32,size(A))
    launch(kernel_log_32,nblocks,1024,(length(A),A,B))
    return B # MEMORY LEAK -- HOW TO REMOVE TEMPORARY B?
end

export log,log!

md=CuModule(DIR*"rectlin.ptx",false)
kernel_rectlin=CuFunction(md,"rectlin")
function rectlin!(A::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_rectlin,nblocks,1024,(length(A),A,B))
end

md=CuModule(DIR*"rectlin_32.ptx",false)
kernel_rectlin_32=CuFunction(md,"rectlin_32")
function rectlin!(A::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_rectlin_32,nblocks,1024,(length(A),A,B))
end
export rectlin!


md=CuModule(DIR*"alex.ptx",false)
kernel_alex=CuFunction(md,"alex")
function alex!(A::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_alex,nblocks,1024,(length(A),A,B))
end

md=CuModule(DIR*"alex_32.ptx",false)
kernel_alex_32=CuFunction(md,"alex_32")
function alex!(A::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_alex_32,nblocks,1024,(length(A),A,B))
end
export alex!


md=CuModule(DIR*"gradalex.ptx",false)
kernel_gradalex=CuFunction(md,"gradalex")
function gradalex!(A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_alex,nblocks,1024,(length(A),A,B,C))
end

md=CuModule(DIR*"gradalex_32.ptx",false)
kernel_gradalex_32=CuFunction(md,"gradalex_32")
function gradalex!(A::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_gradalex_32,nblocks,1024,(length(A),A,B,C))
end
export gradalex!



md=CuModule(DIR*"kinklin.ptx",false)
kernel_kinklin=CuFunction(md,"kinklin")
function kinklin!(A::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_kinklin,nblocks,1024,(length(A),0.25,A,B))
end

md=CuModule(DIR*"kinklin_32.ptx",false)
kernel_kinklin_32=CuFunction(md,"kinklin_32")
function kinklin!(A::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_kinklin_32,nblocks,1024,(length(A),0.25,A,B))
end
export kinklin!


function rectlin(A::CudaArray)
    out=CudaArray(eltype(A),size(A))
    rectlin!(A,out)
    return out
end
export rectlin


function alex(A::CudaArray)
    out=CudaArray(eltype(A),size(A))
    alex!(A,out)
    return out
end
export alex




# A.*(B.>0):
md=CuModule(DIR*"A_emult_Bg0.ptx",false)
kernel_A_emult_Bg0=CuFunction(md,"A_emult_Bg0")
function A_emult_Bg0!(A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_A_emult_Bg0,nblocks,1024,(length(A),A,B,C))
end

md=CuModule(DIR*"A_emult_Bg0_32.ptx",false)
kernel_A_emult_Bg0_32=CuFunction(md,"A_emult_Bg0_32")
function A_emult_Bg0!(A::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_A_emult_Bg0_32,nblocks,1024,(length(A),A,B,C))
end



export A_emult_Bg0!

md=CuModule(DIR*"alphaaxpy.ptx",false)
alphaaxpy=CuFunction(md,"alphaaxpy")

function alphaaxpy!(alpha::Float64,A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(alphaaxpy,nblocks,1024,(length(C),alpha,A,B,C))
end

md=CuModule(DIR*"alphaaxpy_32.ptx",false)
alphaaxpy_32=CuFunction(md,"alphaaxpy_32")

function alphaaxpy!(alpha::Real,A::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(alphaaxpy_32,nblocks,1024,(length(C),Float32(alpha),A,B,C))
end
export alphaaxpy!



md=CuModule(DIR*"alphaax.ptx",false)
alphaax=CuFunction(md,"alphaax")
function alphaax!(alpha::Float64,A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(alphaax,nblocks,1024,(length(C),alpha,A,B,C))
end

md=CuModule(DIR*"alphaax_32.ptx",false)
alphaax_32=CuFunction(md,"alphaax_32")
function alphaax!(alpha::Float32,A::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(alphaax_32,nblocks,1024,(length(C),alpha,A,B,C))
end
export alphaax!



md=CuModule(DIR*"gaxpy.ptx",false)
gaxpy=CuFunction(md,"gaxpy")
function gaxpy!(alpha::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(gaxpy,nblocks,1024,(length(C),alpha,B,C))
end

md=CuModule(DIR*"gaxpy_32.ptx",false)
gaxpy_32=CuFunction(md,"gaxpy_32")
function gaxpy!(alpha::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(gaxpy_32,nblocks,1024,(length(C),alpha,B,C))
end
export gaxpy!


axpy!(alpha::CudaArray,B::CudaArray,C::CudaArray)=gaxpy!(alpha::CudaArray,B::CudaArray,C::CudaArray)


export axpy!


# NB: I've defined this differently from the Base.scale! -- here scale!(a,X) scales X by factor a and overwrites X. This is unfortunate, but I defined this according to the convention that overwrite functions overwrite their last argument.
import Base.scale!
md=CuModule(DIR*"gscale.ptx",false)
kernel_gscale=CuFunction(md,"gscale")
function scale!(alpha::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(kernel_gscale,nblocks,1024,(length(B),alpha,B))
end

md=CuModule(DIR*"gscale_32.ptx",false)
kernel_gscale_32=CuFunction(md,"gscale_32")
function scale!(alpha::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(kernel_gscale_32,nblocks,1024,(length(B),alpha,B))
end
export scale!



md=CuModule(DIR*"diag.ptx",false)
diag_kernel=CuFunction(md,"diag_kernel")
function diag!(A::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(diag_kernel,nblocks,1024,(length(B),A,B))
end

md=CuModule(DIR*"diag_32.ptx",false)
diag_kernel_32=CuFunction(md,"diag_kernel_32")
function diag!(A::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(diag_kernel_32,nblocks,1024,(length(B),A,B))
end
export diag!

md=CuModule(DIR*"diagm.ptx",false)
diagm_kernel=CuFunction(md,"diagm_kernel")
function diagm!(A::CudaArray{Float64},B::CudaArray{Float64})
    nblocks=round(Int,ceil(length(A)/1024))
    fill!(B,0.0)
    launch(diagm_kernel,nblocks,1024,(length(A),A,B))
end


md=CuModule(DIR*"diagm_32.ptx",false)
diagm_kernel_32=CuFunction(md,"diagm_kernel_32")
function diagm!(A::CudaArray{Float32},B::CudaArray{Float32})
    nblocks=round(Int,ceil(length(A)/1024))
    fill!(B,Float32(0.0))
    launch(diagm_kernel_32,nblocks,1024,(length(A),A,B))
end
export diagm!


md=CuModule(DIR*"gax.ptx",false)
gax=CuFunction(md,"gax")
function gax!(alpha::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(gax,nblocks,1024,(length(C),alpha,B,C))
end

md=CuModule(DIR*"gax_32.ptx",false)
gax_32=CuFunction(md,"gax_32")
function gax!(alpha::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(gax_32,nblocks,1024,(length(C),alpha,B,C))
end
export gax!

md=CuModule(DIR*"ax.ptx",false)
ax=CuFunction(md,"ax")
function ax!(alpha::Float64,B::CudaArray{Float64},C::CudaArray{Float64})
       nblocks=round(Int,ceil(length(B)/1024))
    launch(ax,nblocks,1024,(length(C),alpha,B,C))
end


md=CuModule(DIR*"ax_32.ptx",false)
ax_32=CuFunction(md,"ax_32")
function ax!(alpha::Float32,B::CudaArray{Float32},C::CudaArray{Float32})
       nblocks=round(Int,ceil(length(B)/1024))
    launch(ax_32,nblocks,1024,(length(C),alpha,B,C))
end
export ax!

md=CuModule(DIR*"vmultbang.ptx",false)
vmultbangkernel=CuFunction(md,"vmultbang")
function vmult!(alpha::Float64,A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(vmultbangkernel,nblocks,1024,(length(A),alpha,A,B,C))
end

md=CuModule(DIR*"vmultbang_32.ptx",false)
vmultbangkernel_32=CuFunction(md,"vmultbang_32")
function vmult!(alpha::Real,A::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(vmultbangkernel_32,nblocks,1024,(length(A),Float32(alpha),A,B,C))
end
export vmult!

md=CuModule(DIR*"vdivbang.ptx",false)
vdivbangkernel=CuFunction(md,"vdivbang")
function vdiv!(alpha::Float64,A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(vdivbangkernel,nblocks,1024,(length(A),alpha,A,B,C))
end

md=CuModule(DIR*"vdivbang_32.ptx",false)
vdivbangkernel_32=CuFunction(md,"vdivbang_32")
function vdiv!(alpha::Real,A::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
    nblocks=round(Int,ceil(length(B)/1024))
    launch(vdivbangkernel,nblocks,1024,(length(A),Float32(alpha),A,B,C))
end
export vdiv!

md=CuModule(DIR*"vmultbangupdate.ptx",false)
vmultbangupdate_kernel=CuFunction(md,"vmultbangupdate")
function vmultupdate!(alpha::Float64,A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
        nblocks=round(Int,ceil(length(B)/1024))
    launch(vmultbangupdate_kernel,nblocks,1024,(length(A),alpha,A,B,C))
end

md=CuModule(DIR*"vmultbangupdate_32.ptx",false)
vmultbangupdate_kernel_32=CuFunction(md,"vmultbangupdate_32")
function vmultupdate!(alpha::Real,A::CudaArray{Float32},B::CudaArray{Float32},C::CudaArray{Float32})
        nblocks=round(Int,ceil(length(B)/1024))
    launch(vmultbangupdate_kernel_32,nblocks,1024,(length(A),alpha,A,B,C))
end


#md=CuModule(DIR*"vdivbangupdate.ptx",false)
#vdivbangupdate_kernel=CuFunction(md,"vdivbangupdate")
#function vdivupdate!(alpha::Float64,A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64})
#    nblocks=round(Int,ceil(length(B)/1024))
#    launch(vdivbangupdate_kernel,nblocks,1024,(length(A),alpha,A,B,C))
#end

#md=CuModule(DIR*"vAoverBupdate.ptx",false)
#vAoverBupdate_kernel=CuFunction(md,"vAoverBupdate")
#function vAoverBupdate!(alpha::Float64,A::CudaArray{Float64},B::CudaArray{Float64},C::CudaArray{Float64},D::CudaArray{Float64})
#    nblocks=round(Int,ceil(length(B)/1024))
#    launch(vAoverBupdate_kernel,nblocks,1024,(length(A),alpha,A,B,C,D))
#end

function axpy(a::Float64,x::CudaArray{Float64},y::CudaArray{Float64})
    out=copy(y)
    CUBLAS.axpy!(a,x,out)
    return out
end


function axpy(a::Real,x::CudaArray{Float32},y::CudaArray{Float32})
    out=copy(y)
    CUBLAS.axpy!(Float32(a),x,out)
    return out
end
export axpy

flatten(T::Type,g::CudaArray)=reinterpret(T,g,(length(g),1)) # make a (prod(dims),1) CudaArray from a CudaArray
# This is useful since we can then call CUBLAS routines that operate on vectors

flatten(g::CudaArray)=reinterpret(eltype(g),g,(length(g),1)) # make a (prod(dims),1) CudaArray from a CudaArray
# This is useful since we can then call CUBLAS routines that operate on vectors
export flatten

export flatten
