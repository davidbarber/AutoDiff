# f(x)=A*diagX
function FAmultdiagX(A::Array,X::Array)
    if size(A)==(1,1)
        return (A[1].*diagm(vec(X)),nothing)
    elseif size(X)==(1,1)
        return (A.*X[1],nothing)
    else
        return (A*diagm(vec(X)),nothing)
    end
end

function FAmultdiagX_inplace(value,auxvalue,A::Array,X::Array)
    if size(A)==(1,1)
        copy!(value,A[1].*diagm(vec(X)))
    elseif size(X)==(1,1)
        copy!(value,A.*X[1])
    else
        copy!(value,A*diagm(vec(X))) # BLAS has no dgmm routine
    end
end


function DAmultdiagX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X)
    if derivativeIDX==1
        if size(A)==(1,1)
            axpy!(1.0,[sum(X.*diag(grad_c))],grad_n) # sum_i x_i gradc_ii
        elseif size(X)==(1,1)
            axpy!(X[1],grad_c,grad_n) # x gradc_ii
        else
            axpy!(1.0,grad_c.*repmat(X',size(grad_c,1),1),grad_n)
        end
    elseif derivativeIDX==2
        if size(A)==(1,1)
            axpy!(A[1],diag(grad_c),grad_n)
        elseif size(X)==(1,1)
            axpy!(1.0,[sum(A.*grad_c)],grad_n)
        else
            axpy!(1.0,sum(A.*grad_c,1),grad_n)
        end
    end
end

if PROC=="GPU"
    function FAmultdiagX_inplace(value::CudaArray,auxvalue,A::CudaArray,X::CudaArray)
        if size(A)==(1,1)
            diagm!(X,value)
            scale!(A,value)
        elseif size(X)==(1,1)
            copy!(value,A)
            scale!(X,value) # nb argument converse of Base.scale!
        else
            CUBLAS.dgmm!('R',A,vec(X),value)
        end
    end

    function DAmultdiagX(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray)
        if derivativeIDX==1
            if size(A)==(1,1)
                tmp=CudaArray(Float64,(size(X,1),1))
                diag!(grad_c,tmp)
                vmult!(1.0,X,tmp,tmp)
                sum_update!(1.0,tmp,1.0,grad_n)
            elseif size(X)==(1,1)
                alphaaxpy!(1.0,X,grad_c,grad_n) # x gradc_ii
            else
                onc=CudaArray(Float64,(size(grad_c,1),1))
                fill!(onc,1.0);
                tmp=CudaArray(Float64,size(grad_c))
                gemm!('N','T',1.0,onc,X,0.0,tmp)
                vmultupdate!(1.0,grad_c,tmp,grad_n)
                free(tmp); free(onc)
                #                axpy!(1.0,grad_c.*repmat(X',size(grad_c,1),1),grad_n)           
            end
        elseif derivativeIDX==2
            if size(A)==(1,1)
#                axpy!(A[1],diag(grad_c),grad_n)
                tmp=CudaArray(Float64,size(grad_n))
                diag!(grad_c,tmp)
                alphaaxpy!(1.0,A,tmp,grad_n)
                free(tmp)
            elseif size(X)==(1,1)
#                axpy!(1.0,[sum(A.*grad_c)],grad_n)
                tmp=CudaArray(Float64,size(A))
                vmult!(1.0,A,grad_c,tmp)
                sum_update!(1.0,tmp,1.0,grad_n)
                free(tmp)
            else
#                axpy!(1.0,sum(A.*grad_c,1),grad_n)
                tmp=CudaArray(Float64,size(A))
                vmult!(1.0,A,grad_c,tmp)
                tmp2=CudaArray(Float64,(1,size(tmp,2)))
                sum!(tmp,1,0.0,tmp2)
                axpy!(1.0,vec(tmp2),grad_n)
                free(tmp); free(tmp2)
            end
        end
    end
end

Derivative[FAmultdiagX]=DAmultdiagX
export FAmultdiagX

Inplace[FAmultdiagX]=FAmultdiagX_inplace






