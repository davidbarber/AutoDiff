# f(x)=A'*X
function FAtransposeX(A,X)
    if size(A)==(1,1)
        return (A[1].*X,nothing)
    elseif size(X)==(1,1)
        return (A'.*X[1],nothing)
    else
        return (A'*X,nothing)
    end
end

function FAtransposeX_inplace(value,auxvalue,A,X)
    if size(A)==(1,1)
        copy!(value,A[1]*X)
    elseif size(X)==(1,1)
        copy!(value,A'*X[1])
    else
        BLAS.gemm!('T','N',1.0,A,X,0.0,value)
    end
end


function DAtransposeX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X)
    if derivativeIDX==1
        if size(A)==(1,1)
            axpy!(1.0,[sum(X.*grad_c)],grad_n)
        elseif size(X)==(1,1)
            axpy!(X[1],grad_c',grad_n)
        else
            BLAS.gemm!('N','T',1.0,X,grad_c,1.0,grad_n) # X*grad_c'
        end
    elseif derivativeIDX==2
        if size(A)==(1,1)
            axpy!(A[1],grad_c,grad_n)
        elseif size(X)==(1,1)
            axpy!(1.0,[sum(A'.*grad_c)],grad_n)
        else
            BLAS.gemm!('N','N',1.0,A,grad_c,1.0,grad_n) # A*grad_c
        end
    end
end

if PROC=="GPU"
#    function FAtransposeX(A::CudaArray,X::CudaArray)
#        if size(A)==(1,1)
#            value=CudaArray(Float64,size(X));
#            copy!(value,X); scale!(A,value)
#            return (value,[])                        
#        elseif size(X)==(1,1)
#            value=CudaArray(Float64,size(A));
#            copy!(value,A); scale!(X,value) # nb argument converse of Base.scale!
#            return (value,[])                        
#        else
#            return (CUBLAS.gemm('T','N',A,X),[])
#        end
#    end

    function FAtransposeX_inplace(value::CudaArray,auxvalue,A::CudaArray,X::CudaArray)
        if size(A)==(1,1)
            copy!(value,X); scale!(A,value) # nb argument converse of Base.scale!
        elseif size(X)==(1,1)
            CUBLAS.geam!('T','N',1.0,A,0.0,value,value) # nb argument converse of Base.scale!
            scale!(X,value) # nb argument converse of Base.scale!
        else
            CUBLAS.gemm!('T','N',1.0,A,X,0.0,value)
        end
    end

    function DAtransposeX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray)
        if derivativeIDX==1
            if size(A)==(1,1)
                tmp=CudaArray(Float64,size(X))
                vmult!(1.0,X,grad_c,tmp)
                sum_update!(1.0,tmp,1.0,grad_n); free(tmp)
            elseif size(X)==(1,1)
                alphaaxpy!(1.0,X,grad_c,grad_n)
            else
                CUBLAS.gemm!('N','T',1.0,X,grad_c,1.0,grad_n) # X*grad_c'
            end
        elseif derivativeIDX==2
            if size(A)==(1,1)
                alphaaxpy!(1.0,A,grad_c,grad_n)
            elseif size(X)==(1,1)
                tmp=CudaArray(Float64,size(A))
                vmult!(1.0,A,grad_c,tmp)
                sum_update!(1.0,tmp,1.0,grad_n);free(tmp)
            else
                CUBLAS.gemm!('N','N',1.0,A,grad_c,1.0,grad_n) # A*grad_c
            end
        end
    end
end

Derivative[FAtransposeX]=DAtransposeX
export FAtransposeX

Inplace[FAtransposeX]=FAtransposeX_inplace

#import Base.*
#*(A::ADnode,B::ADnode)=ADFunction(FAX,A,B)

#*(A::Real,B::ADnode)=ADFunction(FAX,ADconst(A),B)
#*(A::ADnode,B::Real)=ADFunction(FAX,A,ADconst(B))

#@gpu *(A::CudaArray,B::CudaArray)=CUBLAS.gemm('N','N',A,B)
#export *




