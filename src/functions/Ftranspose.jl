# Transpose is treated in a special way. There are two transpose operations, namely x' and trans(x)
# A' creates a special ADtrans node. The purpose of this is that, for example A'*B can then be recompiled later in to a GEMM call GEMM('T','N',A,B) -- this is efficient since the transpose of A is not actually computed or stored in the graph. These ADtrans nodes are deleted later after optimisation.
#
# On the other hand, trans(A) actually computes and stores the transpose of A on the graph. This should be used sparingly since this will, in general, be very wasteful of storage and computation.

# f(X)=transpose(X)
import Base.transpose
transpose(A::ADnode)=ADFunction(A)
export transpose


ftranspose(A::ADnode)=ADnode(Ftranspose,A)

export ftranpose

Ftranspose(X)=(X',nothing)

function Ftranspose_inplace(value::Array,auxvalue,X::Array)
    copy!(value,X')
end

function Dtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,X::Array)
    axpy!(1.0,grad_c',grad_n)
end


if PROC=="GPU"
    function Ftranspose_inplace(value::CudaArray,auxvalue,X::CudaArray)        
        tmp=CudaArray(Float64,size(X))
        CUBLAS.geam!('T','N',1.0,X,0.0,tmp,value) # value=X'                
        free(tmp)
    end
    
    function Dtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,X::CudaArray)
        tmp=CudaArray(Float64,size(grad_n))
        CUBLAS.geam!('T','N',1.0,grad_c,0.0,tmp,tmp) # tmp=grad_c'                
        axpy!(1.0,tmp,grad_n); free(tmp)
    end

end



Derivative[Ftranspose]=Dtranspose
Inplace[Ftranspose]=Ftranspose_inplace

export Ftranspose



import Base.*

function *(A::ADtrans,B::ADnode)
#    splice!(node,A.index)
    node[A.index]=nothing
    return ADnode(FAtransposeX,[node[A.parent] B])
end

function *(A::ADtrans,B::Real)
    node[A.index]=nothing
    return ADnode(FAtransposeX,[node[A.parent] ADconst(B)])
end

function *(A::ADnode,B::ADtrans)
    node[B.index]=nothing
    return ADnode(FAXtranspose,[A node[B.parent]])
end

function *(A::Real,B::ADtrans)
    node[B.index]=nothing
    return ADnode(FAXtranspose,[ADconst(A) node[B.parent]])
end

function *(A::ADtrans,B::ADtrans)
    node[A.index]=nothing
    node[B.index]=nothing
    return ADnode(FAtransposeXtranspose,[node[A.parent] node[B.parent]])
end

export *

import Base.+

function +(A::ADtrans,B::ADnode)
    node[A.index]=nothing
    return ADnode(FXtransposePY,[node[A.parent] B])
end


function +(A::ADnode,B::ADtrans)
    node[B.index]=nothing
    return ADnode(FXPYtranspose,[A node[B.parent]])
end


function +(A::ADtrans,B::ADtrans)
    node[A.index]=nothing
    node[B.index]=nothing
    return ADnode(FXtransposePYtranspose,[node[A.parent] node[B.parent]])
end


export +


import Base.-

function -(A::ADtrans,B::ADnode)
    node[A.index]=nothing
    return ADnode(FXtransposeMY,[node[A.parent] B])
end

function -(A::ADnode,B::ADtrans)
    node[B.index]=nothing
    return ADnode(FXMYtranspose,[A node[B.parent]])
end

function -(A::ADtrans,B::ADtrans)
    node[A.index]=nothing
    node[B.index]=nothing
    return ADnode(FXtransposeMYtranspose,[node[A.parent] node[B.parent]])
end

trans=ftranspose
export trans

export -

#TODO A'.*B, A.*B', A'.*B'

