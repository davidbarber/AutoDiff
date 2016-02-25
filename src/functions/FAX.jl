# f(x)=A*X
function FAX(A::Array,X::Array)
    if size(A)==(1,1)
        return (A[1].*X,nothing)
    elseif size(X)==(1,1)
        return (A.*X[1],nothing)
    else
        println(A)
        println(X)
        return (A*X,nothing)
    end
end

function FAX_inplace(value,auxvalue,A::Array,X::Array)
    if size(A)==(1,1)
        copy!(value,A[1]*X)
    elseif size(X)==(1,1)
        copy!(value,A*X[1])
    else
        BLAS.gemm!('N','N',1.0,A,X,0.0,value)
    end
end


function DAX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X)
    if derivativeIDX==1
        if size(A)==(1,1)
            axpy!(1.0,[sum(X.*grad_c)],grad_n)
        elseif size(X)==(1,1)
            axpy!(X[1],grad_c,grad_n)
        else
            BLAS.gemm!('N','T',1.0,grad_c,X,1.0,grad_n) # grad_c*X'
        end
    elseif derivativeIDX==2
        if size(A)==(1,1)
            axpy!(A[1],grad_c,grad_n)
        elseif size(X)==(1,1)
            axpy!(1.0,[sum(A.*grad_c)],grad_n)
        else
            BLAS.gemm!('T','N',1.0,A,grad_c,1.0,grad_n) # A'*grad_c
        end
    end
end


if PROC=="GPU"
    gemm!(T1::Char,T2::Char,alpha::Float64,A::CudaArray{Float64},B::CudaArray{Float64},beta::Float64,C::CudaArray{Float64})=CUBLAS.gemm!(T1,T2,alpha,A,B,beta,C)
    gemm!(T1::Char,T2::Char,alpha,A::CudaArray{Float32},B::CudaArray{Float32},beta,C::CudaArray{Float32})=CUBLAS.gemm!(T1,T2,Float32(alpha),A,B,Float32(beta),C)
    export gemm!

    function FAX_inplace(value::CudaArray,auxvalue,A::CudaArray,X::CudaArray)
        if size(A)==(1,1)
            copy!(value,X); scale!(A,value) # nb argument converse of Base.scale!
        elseif size(X)==(1,1)
            copy!(value,A); scale!(X,value) # nb argument converse of Base.scale!
        else
            #CUBLAS.gemm!('N','N',1.0,A,X,0.0,value)
            gemm!('N','N',1.0,A,X,0.0,value)
        end
    end

    function DAX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray)
        if derivativeIDX==1
            if size(A)==(1,1)
                tmp=CudaArray(Float64,size(X))
                vmult!(1.0,X,grad_c,tmp)
                sum_update!(1.0,tmp,1.0,grad_n)
                #axpy!(1.0,sum(tmp),grad_n); 
                free(tmp)
            elseif size(X)==(1,1)
                alphaaxpy!(1.0,X,grad_c,grad_n)
            else
                #CUBLAS.gemm!('N','T',1.0,grad_c,X,1.0,grad_n) # grad_c*X'
                gemm!('N','T',1.0,grad_c,X,1.0,grad_n) # grad_c*X'
            end
        elseif derivativeIDX==2
            if size(A)==(1,1)
                alphaaxpy!(1.0,A,grad_c,grad_n)
            elseif size(X)==(1,1)
                tmp=CudaArray(Float64,size(A))
                vmult!(1.0,A,grad_c,tmp)
                axpy!(1.0,sum(tmp),grad_n); free(tmp)
            else
                #CUBLAS.gemm!('T','N',1.0,A,grad_c,1.0,grad_n) # A'*t
                gemm!('T','N',1.0,A,grad_c,1.0,grad_n) # A'*t
            end
        end
    end



    function DAX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray{Float32},X::CudaArray{Float32})
        if derivativeIDX==1
            if size(A)==(1,1)
                tmp=CudaArray(Float32,size(X))
                vmult!(1.0,X,grad_c,tmp)
                sum_update!(1.0,tmp,1.0,grad_n)
                #axpy!(1.0,sum(tmp),grad_n); 
                free(tmp)
            elseif size(X)==(1,1)
                alphaaxpy!(1.0,X,grad_c,grad_n)
            else
                #CUBLAS.gemm!('N','T',1.0,grad_c,X,1.0,grad_n) # grad_c*X'
                gemm!('N','T',1.0,grad_c,X,1.0,grad_n) # grad_c*X'
            end
        elseif derivativeIDX==2
            if size(A)==(1,1)
                alphaaxpy!(1.0,A,grad_c,grad_n)
            elseif size(X)==(1,1)
                tmp=CudaArray(Float32,size(A))
                vmult!(1.0,A,grad_c,tmp)
                axpy!(1.0,sum(tmp),grad_n); free(tmp)
            else
                #CUBLAS.gemm!('T','N',1.0,A,grad_c,1.0,grad_n) # A'*t
                gemm!('T','N',1.0,A,grad_c,1.0,grad_n) # A'*t
            end
        end
    end

    




    
end

Derivative[FAX]=DAX
export FAX

Inplace[FAX]=FAX_inplace

Derivative[FAX]=DAX
import Base.*
*(A::ADnode,B::ADnode)=ADFunction(FAX,A,B)


*(A::Real,B::ADnode)=ADFunction(FAX,ADconst(A),B)
*(A::ADnode,B::Real)=ADFunction(FAX,A,ADconst(B))


@gpu *(A::CudaArray,B::CudaArray)=CUBLAS.gemm('N','N',A,B)
export *




