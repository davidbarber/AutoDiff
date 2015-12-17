md=CuModule("vcopyshift.ptx",false)
vcopyshift_kernel=CuFunction(md,"vcopyshift")
function copyinto!(out::CudaArray,A::CudaArray,location::Int)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(vcopyshift_kernel,nblocks,1024,(length(A),location-1,A,out))
end
export copyinto!

md=CuModule("vcopyfrom.ptx",false)
vcopyfrom_kernel=CuFunction(md,"vcopyfrom")
function copyfrom!(out::CudaArray,A::CudaArray,location::Int)
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vcopyfrom_kernel,nblocks,1024,(length(out),location-1,A,out))
    end
export copyfrom!

md=CuModule("vcopyfrom_update.ptx",false)
vcopyfrom_update_kernel=CuFunction(md,"vcopyfrom_update")
function copyfrom_update!(out::CudaArray,A::CudaArray,location::Int)
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vcopyfrom_update_kernel,nblocks,1024,(length(out),location-1,A,out))
end
export copyfrom_update!

md=CuModule("vsign.ptx",false)
vsign_kernel=CuFunction(md,"vsign")
function vsign!(A::CudaArray,out::CudaArray)
    nblocks=round(Int,ceil(length(out)/1024))
    launch(vsign_kernel,nblocks,1024,(length(out),A,out))
end
    export vsign!

md=CuModule("gfill.ptx",false)
gfill_kernel=CuFunction(md,"gfill")
function gfill!(out::CudaArray,fillval::CudaArray)
    nblocks=round(Int,ceil(length(out)/1024))
    launch(gfill_kernel,nblocks,1024,(length(out),fillval,out))
end
export gfill!

md=CuModule("tx1mx.ptx",false)
tx1mx_kernel=CuFunction(md,"tx1mx")
function tx1mx!(t::CudaArray,x::CudaArray,out::CudaArray)
    nblocks=round(Int,ceil(length(x)/1024))
    launch(tx1mx_kernel,nblocks,1024,(length(x),t,x,out))
end
export tx1mx!

md=CuModule("Dstanh.ptx",false)
Dstanh_kernel=CuFunction(md,"Dstanh")
function Dstanh!(sf,grad_c::CudaArray,f_c::CudaArray,grad_n::CudaArray)
    nblocks=round(Int,ceil(length(f_c)/1024))
    launch(Dstanh_kernel,nblocks,1024,(length(f_c),sf,grad_c,f_c,grad_n))
    end
export Dstanh!

md=CuModule("DmeanSquareLoss.ptx",false)
DmeanSquareLoss_kernel=CuFunction(md,"DmeanSquareLoss")
function DmeanSquareLoss!(grad_c::CudaArray,x::CudaArray,y::CudaArray,grad_n::CudaArray)
    nblocks=round(Int,ceil(length(x)/1024))
    launch(DmeanSquareLoss_kernel,nblocks,1024,(length(x),2.0/length(x),grad_c,x,y,grad_n))
end
export DmeanSquareLoss!

md=CuModule("stanh.ptx",false)
stanh_kernel=CuFunction(md,"stanh")
function stanh!(sf,A::CudaArray,B::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(stanh_kernel,nblocks,1024,(length(A),sf,A,B))
end
export stanh!

md=CuModule("binaryentropy.ptx",false)
binaryentropy_kernel=CuFunction(md,"binaryentropy")
function binaryentropy(X::CudaArray,Y::CudaArray)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropy_kernel,nblocks,1024,(length(X),X,Y,tmp))
    out=mean(tmp)
    free(tmp)
    return out
end
export binaryentropy

md=CuModule("binaryentropyXsigmoidY.ptx",false)
binaryentropyXsigmoidY_kernel=CuFunction(md,"binaryentropyXsigmoidY")
function binaryentropyXsigmoidY(X::CudaArray,Y::CudaArray)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropyXsigmoidY_kernel,nblocks,1024,(length(X),X,Y,tmp))
    out=mean(tmp)
    free(tmp)
    return out
end
export binaryentropyXsigmoidY

function binaryentropyXsigmoidY!(X::CudaArray,Y::CudaArray,out)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(binaryentropyXsigmoidY_kernel,nblocks,1024,(length(X),X,Y,tmp))
    out=mean(tmp)
    free(tmp)
end
export binaryentropyXsigmoidY!

md=CuModule("DXbinaryentropy.ptx",false)
DXbinaryentropy_kernel=CuFunction(md,"DXbinaryentropy")
function DXbinaryentropy!(X::CudaArray,Y::CudaArray,T::CudaArray,Out::CudaArray)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DXbinaryentropy_kernel,nblocks,1024,(length(X),X,Y,T,Out))
end
export DXbinaryentropy!

md=CuModule("DYbinaryentropy.ptx",false)
DYbinaryentropy_kernel=CuFunction(md,"DYbinaryentropy")
function DYbinaryentropy!(X::CudaArray,Y::CudaArray,T::CudaArray,Out::CudaArray)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DYbinaryentropy_kernel,nblocks,1024,(length(X),X,Y,T,Out))
end
export DYbinaryentropy!

md=CuModule("DXbinaryentropyXsigmoidY.ptx",false)
DXbinaryentropyXsigmoidY_kernel=CuFunction(md,"DXbinaryentropyXsigmoidY")
function DXbinaryentropyXsigmoidY!(X::CudaArray,Y::CudaArray,T::CudaArray,Out::CudaArray)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DXbinaryentropyXsigmoidY_kernel,nblocks,1024,(length(X),X,Y,T,Out))
end
export DXbinaryentropyXsigmoidY!

md=CuModule("DYbinaryentropyXsigmoidY.ptx",false)
DYbinaryentropyXsigmoidY_kernel=CuFunction(md,"DYbinaryentropyXsigmoidY")
function DYbinaryentropyXsigmoidY!(X::CudaArray,Y::CudaArray,T::CudaArray,Out::CudaArray)
    tmp=CudaArray(Float64,size(X))
    nblocks=round(Int,ceil(length(X)/1024))
    launch(DYbinaryentropyXsigmoidY_kernel,nblocks,1024,(length(X),X,Y,T,Out))
end
export DYbinaryentropyXsigmoidY!

md=CuModule("sigmoid.ptx",false)
sigmoid_kernel=CuFunction(md,"sigmoid")
function sigmoid(A::CudaArray)
    out=CudaArray(Float64,size(A))
    nblocks=round(Int,ceil(length(A)/1024))
    launch(sigmoid_kernel,nblocks,1024,(length(A),A,out))
    return out
end
export sigmoid

function sigmoid!(A::CudaArray,out::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(sigmoid_kernel,nblocks,1024,(length(A),A,out))
end
export sigmoid!

md=CuModule("vsquare.ptx",false)
vsquarekernel=CuFunction(md,"vsquare")
function vsquare(A::CudaArray)
    out=zeros(A)
    launch(vsquarekernel,size(A,1),size(A,2),(A,out))
    return out
end

function vsquareout!(A::CudaArray,Out::CudaArray)
    launch(vsquarekernel,size(A,1),size(A,2),(A,Out))
end

import Base.exp
md=CuModule("exp.ptx",false)
kernel_exp=CuFunction(md,"expkernel")
function exp!(A::CudaArray,B::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_exp,nblocks,1024,(length(A),A,B))
end

function exp(A::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    B=CudaArray(Float64,size(A))
    launch(kernel_exp,nblocks,1024,(length(A),A,B))
    return B # MEMORY LEAK -- HOW TO REMOVE TEMPORARY B?
end
export exp,exp!

import Base.log
md=CuModule("log.ptx",false)
kernel_log=CuFunction(md,"logkernel")
function log!(A::CudaArray,B::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_log,nblocks,1024,(length(A),A,B))
end

function log(A::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    B=CudaArray(Float64,size(A))
    launch(kernel_log,nblocks,1024,(length(A),A,B))
    return B # MEMORY LEAK -- HOW TO REMOVE TEMPORARY B?
end
export log,log!

md=CuModule("rectlin.ptx",false)
kernel_rectlin=CuFunction(md,"rectlin")
function rectlin!(A::CudaArray,B::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_rectlin,nblocks,1024,(length(A),A,B))
end
export rectlin!

function rectlin(A::CudaArray)
    out=CudaArray(Float64,size(A))
    rectlin!(A,out)
    return out
end
export rectlin

# A.*(B.>0):
md=CuModule("A_emult_Bg0.ptx",false)
kernel_A_emult_Bg0=CuFunction(md,"A_emult_Bg0")
function A_emult_Bg0!(A::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    launch(kernel_A_emult_Bg0,nblocks,1024,(length(A),A,B,C))
end
export A_emult_Bg0!

md=CuModule("alphaaxpy.ptx",false)
alphaaxpy=CuFunction(md,"alphaaxpy")

function alphaaxpy!(alpha::Float64,A::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(alphaaxpy,nblocks,1024,(length(C),alpha,A,B,C))
end
export alphaaxpy!

md=CuModule("alphaax.ptx",false)
alphaax=CuFunction(md,"alphaax")
function alphaax!(alpha::Float64,A::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(alphaax,nblocks,1024,(length(C),alpha,A,B,C))
end
export alphaax!

md=CuModule("gaxpy.ptx",false)
gaxpy=CuFunction(md,"gaxpy")
function gaxpy!(alpha::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(gaxpy,nblocks,1024,(length(C),alpha,B,C))
end
export gaxpy!


axpy!(alpha::CudaArray,B::CudaArray,C::CudaArray)=gaxpy!(alpha::CudaArray,B::CudaArray,C::CudaArray)
export axpy!


# NB: I've defined this differently from the Base.scale! -- here scale!(a,X) scales X by factor a and overwrites X. This is unfortunate, but I defined this according to the convention that overwrite functions overwrite their last argument.
import Base.scale!
md=CuModule("gscale.ptx",false)
kernel_gscale=CuFunction(md,"gscale")
function scale!(alpha::CudaArray,B::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(kernel_gscale,nblocks,1024,(length(B),alpha,B))
end
export scale!

md=CuModule("diag.ptx",false)
diag_kernel=CuFunction(md,"diag_kernel")
function diag!(A::CudaArray,B::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(diag_kernel,nblocks,1024,(length(B),A,B))
end
export diag!

md=CuModule("diagm.ptx",false)
diagm_kernel=CuFunction(md,"diagm_kernel")
function diagm!(A::CudaArray,B::CudaArray)
    nblocks=round(Int,ceil(length(A)/1024))
    fill!(B,0.0)
    launch(diagm_kernel,nblocks,1024,(length(A),A,B))
end
export diagm!


md=CuModule("gax.ptx",false)
gax=CuFunction(md,"gax")
function gax!(alpha::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(gax,nblocks,1024,(length(C),alpha,B,C))
end
export gax!

md=CuModule("ax.ptx",false)
ax=CuFunction(md,"ax")
function ax!(alpha::Float64,B::CudaArray,C::CudaArray)
       nblocks=round(Int,ceil(length(B)/1024))
    launch(ax,nblocks,1024,(length(C),alpha,B,C))
end
export ax!

md=CuModule("vmultbang.ptx",false)
vmultbangkernel=CuFunction(md,"vmultbang")
function vmult!(alpha::Float64,A::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(vmultbangkernel,nblocks,1024,(length(A),alpha,A,B,C))
end
export vmult!

md=CuModule("vdivbang.ptx",false)
vdivbangkernel=CuFunction(md,"vdivbang")
function vdiv!(alpha::Float64,A::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(vdivbangkernel,nblocks,1024,(length(A),alpha,A,B,C))
end
export vdiv!

md=CuModule("vmultbangupdate.ptx",false)
vmultbangupdate_kernel=CuFunction(md,"vmultbangupdate")
function vmultupdate!(alpha::Float64,A::CudaArray,B::CudaArray,C::CudaArray)
        nblocks=round(Int,ceil(length(B)/1024))
    launch(vmultbangupdate_kernel,nblocks,1024,(length(A),alpha,A,B,C))
end

md=CuModule("vdivbangupdate.ptx",false)
vdivbangupdate_kernel=CuFunction(md,"vdivbangupdate")
function vdivupdate!(alpha::Float64,A::CudaArray,B::CudaArray,C::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(vdivbangupdate_kernel,nblocks,1024,(length(A),alpha,A,B,C))
end

md=CuModule("vAoverBupdate.ptx",false)
vAoverBupdate_kernel=CuFunction(md,"vAoverBupdate")
function vAoverBupdate!(alpha::Float64,A::CudaArray,B::CudaArray,C::CudaArray,D::CudaArray)
    nblocks=round(Int,ceil(length(B)/1024))
    launch(vAoverBupdate_kernel,nblocks,1024,(length(A),alpha,A,B,C,D))
end

function axpy(a::Float64,x::CudaArray,y::CudaArray)
    out=copy(y)
    CUBLAS.axpy!(a,x,out)
    return out
end
export axpy

flatten(T::Type,g::CudaArray)=reinterpret(T,g,(length(g),1)) # make a (prod(dims),1) CudaArray from a CudaArray
# This is useful since we can then call CUBLAS routines that operate on vectors
export flatten
