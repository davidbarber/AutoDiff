# Diag is treated in a special way.

# f(X)=diagm(X)
import Base.diagm
diagm(A::ADnode)=ADdiag(A)
export diagm

#TODO defined diagonalm to construct explicitly a node

#ftranspose(A::ADnode)=ADnode(Ftranspose,A)
#export ftranpose

#Ftranspose(X)=(X',nothing)

#function Ftranspose_inplace(value::Array,auxvalue,X::Array)
#    copy!(value,X')
#end

#function Dtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,X::Array)
#    axpy!(1.0,grad_c',grad_n)
#end


#if PROC=="GPU"
#    function Ftranspose_inplace(value::CudaArray,auxvalue,X::CudaArray)
#        tmp=CudaArray(Float64,size(X))
#        CUBLAS.geam!('T','N',1.0,X,0.0,tmp,value) # value=X'
#        free(tmp)
#    end
#
#    function Dtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,X::CudaArray)
#        tmp=CudaArray(Float64,size(grad_n))
#        CUBLAS.geam!('T','N',1.0,grad_c,0.0,tmp,tmp) # tmp=grad_c'
#        axpy!(1.0,tmp,grad_n); free(tmp)
#    end
#
#end


#Derivative[Ftranspose]=Dtranspose
#Inplace[Ftranspose]=Ftranspose_inplace
#
#export Ftranspose

import Base.*

function *(A::ADdiag,B::ADnode)
    node[A.index]=nothing
    return ADnode(FdiagAmultX,[node[A.parent] B])
end

function *(A::ADdiag,B::Real)
    node[A.index]=nothing
    return diagm(node[A.parent]*ADconst(B))
end

function *(A::ADnode,B::ADdiag)
    node[B.index]=nothing
    return ADnode(FAmultdiagX,[A node[B.parent]])
end

function *(A::Real,B::ADdiag)
    node[B.index]=nothing
    return diagm(ADconst(A)*node[B.parent])
end


function *(A::ADdiag,B::ADdiag)
    node[A.index]=nothing
    node[B.index]=nothing
    return diagm(node[A.parent].*node[B.parent])
end


export *

#import Base.+#

#function +(A::ADtrans,B::ADnode)
#    node[A.index]=nothing
#    return ADnode(FXtransposePY,[node[A.parent] B])
#end


#function +(A::ADnode,B::ADtrans)
#    node[B.index]=nothing
#    return ADnode(FXPYtranspose,[A node[B.parent]])
#end


#function +(A::ADtrans,B::ADtrans)
#    node[A.index]=nothing
#    node[B.index]=nothing
#    return ADnode(FXtransposePYtranspose,[node[A.parent] node[B.parent]])
#end


#export +


#import Base.-

#function -(A::ADtrans,B::ADnode)
#    node[A.index]=nothing
#    return ADnode(FXtransposeMY,[node[A.parent] B])
#end

#function -(A::ADnode,B::ADtrans)
#    node[B.index]=nothing
#    return ADnode(FXMYtranspose,[A node[B.parent]])
#end

#function -(A::ADtrans,B::ADtrans)
#    node[A.index]=nothing
#    node[B.index]=nothing
#    return ADnode(FXtransposeMYtranspose,[node[A.parent] node[B.parent]])
#end

#trans=ftranspose
#export trans

#export -



