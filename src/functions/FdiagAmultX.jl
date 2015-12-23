# f(x)=diagA*X
function FdiagAmultX(A::Array,X::Array)
    if size(A)==(1,1)
        return (A[1].*X,nothing)
    elseif size(X)==(1,1)
        return (diagm(vec(A)).*X[1],nothing)
    else
        return (diagm(vec(A))*X,nothing)
    end
end

function FdiagAmultX_inplace(value,auxvalue,A::Array,X::Array)
    if size(A)==(1,1)
        copy!(value,A[1]*X)
    elseif size(X)==(1,1)
        copy!(value,diagm(vec(A))*X[1]) 
    else
        copy!(value,diagm(vec(A))*X) # BLAS has no dgmm routine
    end
end

function DdiagAmultX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X)
    if derivativeIDX==1
        if size(A)==(1,1)
            axpy!(1.0,[sum(X.*grad_c)],grad_n)
        elseif size(X)==(1,1)
            axpy!(X[1],sum(grad_c,2),grad_n)
        else
            axpy!(1.0,sum(X.*grad_c,2),grad_n)
        end
    elseif derivativeIDX==2
        if size(A)==(1,1)
            axpy!(A[1],grad_c,grad_n)
        elseif size(X)==(1,1)
            #axpy!(1.0,[sum(A.*diag(grad_c))],grad_n) 
            axpy!(1.0,[sum(sum(A,2).*diag(grad_c))],grad_n)
        else
            axpy!(1.0,diagm(vec(A))*grad_c,grad_n)
        end
    end
end

if PROC=="GPU"
    function FdiagAmultX_inplace(value::CudaArray,auxvalue,A::CudaArray,X::CudaArray)
        if size(A)==(1,1)
            gax!(A,X,value)
        elseif size(X)==(1,1)
            diagm!(A,value)
            scale!(X,value) # nb argument converse of Base.scale!
        else
            CUBLAS.dgmm!('L',X,vec(A),value)
        end
    end

    function DdiagAmultX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray)
        if derivativeIDX==1
            if size(A)==(1,1)
                tmp=CudaArray(Float64,size(X))
                vmult!(1.0,X,grad_c,tmp)
                sum_update!(1.0,tmp,1.0,grad_n)
                free(tmp)
            elseif size(X)==(1,1)
                tmp=CudaArray(Float64,size(grad_n))
                sum!(grad_c,2,0.0,tmp) # tmp=sum(grad_c,2)
                alphaaxpy!(1.0,X,tmp,grad_n)
                free(tmp)
            else
                tmp=CudaArray(Float64,size(X))
                vmult!(1.0,X,grad_c,tmp) #tmp=X.*grad_c
                sum!(tmp,2,1.0,grad_n)
            end
        elseif derivativeIDX==2
            if size(A)==(1,1)
                alphaaxpy!(1.0,A,grad_c,grad_n)
            elseif size(X)==(1,1)
              tmp1=CudaArray(Float64,(size(A,1),1))
              sum!(A,2,0.0,tmp1) # sum(A,2)
              tmp2=CudaArray(Float64,(size(A,1),1))
              diag!(grad_c,tmp2)
              vmult!(1.0,tmp1,tmp2,tmp1)
              sum_update!(1.0,tmp1,1.0,grad_n)
              free(tmp1); free(tmp2)
#            axpy!(1.0,[sum(sum(A,2).*diag(grad_c))],grad_n)
            else
                tmp=CudaArray(Float64,size(grad_c))
                             CUBLAS.dgmm!('L',grad_c,vec(A),tmp)
                axpy!(1.0,tmp,grad_n)
                free(tmp)
            end
        end
    end
end

Derivative[FdiagAmultX]=DdiagAmultX
export FdiagAmultX

Inplace[FdiagAmultX]=FdiagAmultX_inplace






