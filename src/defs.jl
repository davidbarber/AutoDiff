# (c) David Barber, University College London 2015
#= How this works:

GENERAL AUTODIFF THEORY:

According to the general autodiff theory (see my online notes http://web4.cs.ucl.ac.uk/staff/D.Barber/publications/ParameterTying.pdf) the total derivative t[n] at a node n in the graph is related to the children c of node n by
 \sum_c t[c] df[c]/df[n]

or, in the notation used in the code

grad_n = \sum_c grad_c* df[c]/df[n]

RETURNING A TOTAL DERIVATIVE FUNCTION:

Consider a node that computes h=sigma(W*x).  Since the output of this node h is a vector, we need to calculate all the derivatives from these elements of this child h to its parents W and x. For example,

dh[i]/dW[a,b] = dsigma(W[a,:]'*x)*x[b]*delta[i,a]

where delta[i,a] is the Kronecker delta function and dsigma is the derivative of the transfer function sigma. However, when we use this as part of the reverse recursion, the Kronecker delta gives a simplification, namely

\sum_i grad_c[i]dh[i]/dW[a,b] = grad_c[a]dsigma(W[a,:]'*x)*x[b]

What this means is that we can exploit the mathematical simplication that results from the Kronecker delta and return instead a derivative as a function of the child total derivative grad_c. For example, for the logistic sigmoid, dsigma=sigma*(1-sigma) and we use

(grad_c.*f_c.*(1.-f_c))*x'   for the derivative wrt W
W'*(grad_c.*f_c.*(1.-f_c))   for the derivative wrt x

Here f_c is the value of the child node (which is sigma(W*x).

The derivative can then be returned directly as a function of the node value, the parent node values and the child node value -- no additional storage is required. This is more efficient than defining two nodes, first h=W*x, then sigma(h); here we would have to store the value h and also sigma(h), which will cost roughly twice as much in terms of storage.

REUSING INFORMATION BY STORING AUXILIARY VALUES:

Some computational savings can be made in the reverse pass by reusing results computed during the forward pass.

For example, for the logistic sigmoid, we calculate f=sigma(W*h) on the forward pass. The derivative pass requires us to calculate dsigma(W*x) but since this is equal to sigma(W*h)*(1-sigma(W*h)) this is just a simple function of the forward pass result, namely f.*(1-f) so that we can simply reuse the forward calculation values, without having to compute the sigmoid function again in the reverse pass -- this saves on computation.

The code therefore allows during the forward pass to store auxilliary information at a node that might be useful for speeding up the computation in the backward pass. The deriviate can then be a function of the forward pass value (f_c) or the auxilliary information (faux_c) stored at the forward pass node.

Note that the functions all have two versions -- a standard one which returns a value, and an inplace version that updates an existing value. (The CUDA versions only require the inplace routines).  The standard one is only required during compilation. All subsequent uses of the function are inplace.

The gradients are defined to update preexisting gradients -- inplace. It's also important to bear in mind that (at each backward pass, after iniitalisating gradients to zero) we must add gradients at a node. An explanation for this is that we can have situations in which a variable x has two directed paths to a variable f. For example, f=sum([x x]).



For future, it would be interesting to map any function f and reduce it with sum

function Fsumf(f,x...)
    return sum(f(collect(x)))
end
Then
Dsumf(derivativeIDX,s,a,t,grad,reset,D,f,x...)=t.*D[f](x[derivativeIDX])
We could even then find the symbolic derivative df of f. We wouldn't want to have to do this each time we call the function though.
We could I think do this with a Dict.
    D=Dict()
    D["x^2"]=(x)->2x
    We can then evaluate using
    D["x^2"](3.2)
    We can then make the Dict and the function f both nodes in the graph
    For the GPU though we would need to have compiled kernels for each function f and its associated derivative df
=#

# --------------------------------------------------------------------------------------
# Functions and their Derivatives:

# functions are defined as F(x)=(self,aux). Here self is the function value and aux is the auxiliary value (it can be empty []) that might be useful to speed up the return pass calculation.

Derivative=Dict() # mappings of functions to their derivatives
Inplace=Dict() # Inplace version of function


#TODO:
# sqrt, power fn, trig functions,
# diagm(A)+
# diag(A) for both CPU and GPU (note that diagm(A) is implemented for both GPU and CPU)
# A/s where s is a 1x1 array and A is an array
# A.*B', A'.*B, A'.*B' for CPU and GPU; currently these are computed using A.*trans(B), etc which is wasteful
# A./B', A'./B, A'./B' for CPU and GPU
# array subindexing (see old code at end) Not sure if this is really necessary
# The KL loss methods are expensive if one only wants to maximise the corresponding likelihood. Could make versions with a flag to drop the entropy term
# Some of the GPU demos are very slow -- DemoLSTMsimple, for example. Need to investigate why.

f=[
   "Fnx",
   "Fstanh", "Fsigmoid", "Frectlin",
   "FAhadamardprodB",    "FAhadamarddivB",
   "Fsum", "Fmean",
   "FsumSquare",    "FmeanSquare",
   "FAX", "FAXplusBias",
   "Fxpy", "Fxmy",
   "FmeanSquareLoss",
   "FBinaryKullbackLeiblerLoss", "FBinaryKullbackLeiblerLossXsigmoidY",
   "FsigmoidAXplusBias",
   "FrectlinAXplusBias",
   "FstanhAXplusBias",
   "Fsoftmax",   "FKLsoftmax",
   "Fexp", "Flog",
   "Ftranspose",
   "FAtransposeX",
   "FAXtranspose",
   "FAtransposeXtranspose",
   "FXtransposePY",
   "FXPYtranspose",
   "FXtransposePYtranspose",
   "FXtransposeMY",
   "FXMYtranspose",
   "FXtransposeMYtranspose",
   "FmeanAbs","Fabs",
   "Fvcat", "Fhcat",
   "Fdiagm", "FdiagAmultX", "FAmultdiagX",
   "FabsAXplusBias",
   "Felu",
   "Fabs",
   "Fkinklin",
   "FkinklinAXplusBias",
    "Falex"
   "FConvolution",
   "FCUrectlin",
   "FCUsoftmax",
   "FPooling",
   "FActivation"
   ]

for fn in f
    include("functions/"*fn*".jl");  println(fn);
end

include("TrainingAlgorithms.jl")


## Negative function: f(x)=-x
#Fnegative(x)=(-x,[])
#Dnegative=Array(Function,1)
#Dnegative[1]=dnegative1(x,self,aux,t)=-t.*ones(size(x))
#Dnegative[1]=dnegative1(x::Float32,self,aux,t)=-t
#Derivative[Fnegative]=Dnegative # Define dictionary lookup
#export Fnegative
#export dnegative1 # need for source code execution
#ADnegative(n)=ADnode(Fnegative,n)
#export ADnegative

## KL Loss: f(p,q)=KL(p,q)
#FKLLoss(p::Array{Float32,2},q::Array{Float32,2})=begin DN=prod(size(p));(sum(p.*(log(p)-log(q)))/DN,DN); end
#DKLLoss=Array(Function,2)
#DKLLoss[1]=DKLLoss1(p::Array{Float32,2},q::Array{Float32,2},self,aux,t)=t.*(1+log(p)-log(q))/aux
#DKLLoss[2]=DKLLoss2(p::Array{Float32,2},q::Array{Float32,2},self,aux,t)=t.*(p./q)./aux
#Derivative[FKLLoss]=DKLLoss
#export FKLLoss,DKLLoss
#export DKLLoss1,DKLLoss2 # need for source code execution
#ADKLLoss(np,nq)=ADnode(FKLLoss,[np nq])
#export ADKLLoss
#
# Multinomial Logistic Loss: -sum(log(p_c)) where p\propto exp(x) and c is a bit array of the same size as x
#FMultLogisticLoss(c::BitArray{2},x::Array{Float32,2})=begin DN=prod(size(x));logZ=logsumexp(x);aux=(logZ,DN); return (#(-sum(x[c])+sum(logZ))/DN,aux); end
#DMultLogisticLoss=Array(Function,2)
#DMultLogisticLoss[1]=DMultLogisticLoss1(c::BitArray{2},x::Array{Float32,2},self,aux,t)=nan # not needed
#DMultLogisticLoss[2]=DMultLogisticLoss2(c::BitArray{2},x::Array{Float32,2},self,aux,t)=begin p=zeros(size(c)); p[c]=1.#; return t.*(exp(x.-aux[1])-p)./aux[2]; end
#Derivative[FMultLogisticLoss]=DMultLogisticLoss
#export FMultLogisticLoss,DMultLogisticLoss
#export DMultLogisticLoss1,DMultLogisticLoss2 # need for source code execution
#ADMultLogisticLoss(nc,nx)=ADnode(FMultLogisticLoss,[nc nx])
#export ADMultLogisticLoss
#
##Logistic Loss: -sum(log(sigma(c.*x)))/prod(size(x)) c[i] is +1 or -1 class variable
#FLogisticLoss(c::Array{Float32,2},x::Array{Float32,2})=begin DN=prod(size(x)); aux=(1./(1.+exp(-c.*x)),DN); return (-s#um(log(aux[1]))/DN,aux); end
#DLogisticLoss=Array(Function,2)
#DLogisticLoss[1]=DLogisticLoss1(c::Array{Float32,2},x::Array{Float32,2},self,aux,t)=-t.*(1.-aux[1]).*x/aux[2]
#DLogisticLoss[2]=DLogisticLoss2(c::Array{Float32,2},x::Array{Float32,2},self,aux,t)=-t.*(1.-aux[1]).*c/aux[2]
#
#FLogisticLoss(c::Array{Float32},x::Array{Float32})=begin DN=prod(size(x)); aux=(1./(1.+exp(-c.*x)),DN); return (-sum(l#og(aux[1]))/DN,aux); end
#DLogisticLoss=Array(Function,2)
#DLogisticLoss[1]=DLogisticLoss1(c::Array{Float32,1},x::Array{Float32,2},self,aux,t)=-t.*(1.-aux[1]).*x/aux[2]
#DLogisticLoss[2]=DLogisticLoss2(c::Array{Float32,1},x::Array{Float32,2},self,aux,t)=-t.*(1.-aux[1]).*c/aux[2]
#Derivative[FLogisticLoss]=DLogisticLoss
#export FLogisticLoss,DLogisticLoss
#export DLogisticLoss1,DLogisticLoss2 # need for source code execution
#ADLogisticLoss(nc,nx)=ADnode(FLogisticLoss,[nc nx])
#export ADLogisticLoss


## Array subindexing:
#Fgetcol(x,col)=begin if col>0 tmp=zeros(size(x,1),1); tmp[:]=x[:,col]; return (tmp, zeros(size(x))) else retur#n (zeros(size(x,1)), zeros(size(x))); end; end
#Dgetcol=Array(Function,2)
#Dgetcol[1]=dgetcol1(x,col,self,aux,t)=begin if col>0 aux[:,col]=t; return aux; else return zeros(size(x)); end#; end
#Dgetcol[2]=dgetcol2(x,col,self,aux,t)=[] # can't differentiate wrt col index
#Derivative[Fgetcol]=Dgetcol
#export Fgetcol
#export dgetcol1,dgetcol2 # need for source code execution
#ADgetcol(n,col)=ADnode(Fgetcol,[n col])
#export ADgetcol


#    md=CuModule("gaxpy2.ptx",false)
#    gaxpy2=CuFunction(md,"gaxpy2")
#    function gaxpy2!(alpha::CudaArray,B::CudaArray,C::CudaArray)
#        launch(gaxpy2,size(B,1),size(B,2),(alpha,B,C))
#    end
#    export gaxpy2!

#    md=CuModule("gaxpy3.ptx",false)
#    gaxpy3=CuFunction(md,"gaxpy3")
#    function gaxpy3!(alpha::CudaArray,B::CudaArray,C::CudaArray)
#        launch(gaxpy3,(size(B,1),size(B,2)),1,(length(B),alpha,B,C))
#    end
#    export gaxpy3!

#    md=CuModule("gaxpy4.ptx",false)
#    gaxpy4=CuFunction(md,"gaxpy4")
#    function gaxpy4!(alpha::CudaArray,B::CudaArray,C::CudaArray)
#        launch(gaxpy4,(size(B,1),size(B,2)),1024,(length(B),alpha,B,C))
#    end
#    export gaxpy4!
#    export gaxpy4


#    md=CuModule("serialsum.ptx",false)
#    serialsumkernel=CuFunction(md,"serialsum")
#    function serialsum(A::CudaArray)
#        gy=CudaArray(ones(1,1))
#        launch(serialsumkernel,1,1,(length(A),A,gy))
#        return gy
#    end
#    export serialsum

