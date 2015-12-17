# f(x)=A'*X'
function FAtransposeXtranspose(A,X)
    if size(A)==(1,1)
        return (A[1].*X',nothing)
    elseif size(X)==(1,1)
        return (A'.*X[1],nothing)
    else
        return (A'*X',nothing)
    end
end

function FAtransposeXtranspose_inplace(value,auxvalue,A,X)
    if size(A)==(1,1)
        copy!(value,A[1]*X')
    elseif size(X)==(1,1)
        copy!(value,A'*X[1])
    else
        BLAS.gemm!('T','T',1.0,A,X,0.0,value)
    end
end


function DAtransposeXtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X)
    if derivativeIDX==1
        if size(A)==(1,1)
            axpy!(1.0,[sum(X'.*grad_c)],grad_n)
        elseif size(X)==(1,1)
            axpy!(X[1],grad_c',grad_n)
        else
            BLAS.gemm!('T','T',1.0,X,grad_c,1.0,grad_n) # X'*grad_c'
        end
    elseif derivativeIDX==2
        if size(A)==(1,1)
            axpy!(A[1],grad_c,grad_n)
        elseif size(X)==(1,1)
            axpy!(1.0,[sum(A'.*grad_c)],grad_n)
        else
            BLAS.gemm!('T','T',1.0,grad_c,A,1.0,grad_n) # grad_c'*A'
        end
    end
end



if GPU
#    function FAtransposeXtranpose(A::CudaArray,X::CudaArray)
#        if size(A)==(1,1)
#            tmp=CudaArray(Float64,size(X)); gfill!(tmp,A)
#            return (CUBLAS.gemm('N','N',tmp,X),nothing)
#        elseif size(X)==(1,1)
#            tmp=CudaArray(Float64,size(A)); gfill!(tmp,X)
#            return (CUBLAS.gemm('N','N',A,tmp),nothing)
#        else
#            return (CUBLAS.gemm('N','N',A,X),[])
#        end
#    end

    function FAtransposeXtranspose_inplace(value::CudaArray,auxvalue,A::CudaArray,X::CudaArray)
        if size(A)==(1,1)
            CUBLAS.geam!('T','N',1.0,X,0.0,value,value) # nb argument converse of Base.scale!
            scale!(A,value) # nb argument converse of Base.scale!
        elseif size(X)==(1,1)
            CUBLAS.geam!('T','N',1.0,A,0.0,value,value) # nb argument converse of Base.scale!
            scale!(X,value) # nb argument converse of Base.scale!
        else
            CUBLAS.gemm!('T','T',1.0,A,X,0.0,value)
        end
    end

    function DAtransposeXtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray)
        if derivativeIDX==1
            if size(A)==(1,1)
                tmp=CudaArray(Float64,size(X))
                CUBLAS.geam!('T','N',1.0,X,0.0,tmp,tmp) # tmp=X'                
                vmult!(1.0,tmp,grad_c,tmp) # tmp=X'.*grad_c
                sum_update!(1.0,tmp,1.0,grad_n) # grad_n+=sum(X'.*grad_c)
                free(tmp)
            elseif size(X)==(1,1)
                alphaaxpy!(1.0,X,grad_c,grad_n)
            else
                CUBLAS.gemm!('T','T',1.0,X,grad_c,1.0,grad_n) # X'*grad_c'
            end
        elseif derivativeIDX==2
            if size(A)==(1,1)
                tmp=CudaArray(Float64,size(grad_c))
                CUBLAS.geam!('T','N',1.0,grad_c,0.0,tmp,tmp) # tmp=grad_c'                
                alphaaxpy!(1.0,A,tmp,grad_n); free(tmp)
            elseif size(X)==(1,1)
                tmp=CudaArray(Float64,size(A))
                vmult!(1.0,A,grad_c,tmp)
                sum_update!(1.0,tmp,1.0,grad_n);free(tmp)

            else
                CUBLAS.gemm!('T','T',1.0,grad_c,A,1.0,grad_n) # grad_c'*A'
            end
        end
    end
end

Derivative[FAtransposeXtranspose]=DAtransposeXtranspose
export FAtransposeXtranspose

Inplace[FAtransposeXtranspose]=FAtransposeXtranspose_inplace



