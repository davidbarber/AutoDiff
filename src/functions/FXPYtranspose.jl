# f(x,y)=x+y'

FXPYtranspose(x::Float64,y::Float64)=Fxpy(x,y)
FXPYtranspose(x::Float64,y)=Fxpy(x,y')

FXPYtranspose(x,y::Float64)=Fxpy(x,y)

FXPYtranspose(x::Array,y::Array)=Fxpy(x,y')

FXPYtranspose_inplace(value,auxvalue,x::Array,y::Array)=Fxpy_inplace(value,auxvalue,x,y')

function DXPYtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::Array,y::Array)
    if derivativeIDX==1
        if size(x)==(1,1)
            axpy!(1.0,[sum(grad_c)],grad_n)
        else
            axpy!(1.0,grad_c,grad_n)
        end
    elseif derivativeIDX==2
        if size(y)==(1,1)
            axpy!(1.0,[sum(grad_c)],grad_n)
        else
            axpy!(1.0,grad_c',grad_n)
        end
    end
end

if PROC=="GPU"

FXPYtranspose_inplace(value,auxvalue,x::CudaArray,y::CudaArray)=FXtransposePY_inplace(value,auxvalue,y,x)

function DXPYtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
    if derivativeIDX==1
        if size(x)==(1,1)
            axpy!(1.0,sum(grad_c),grad_n)
        else
            axpy!(1.0,grad_c,grad_n)
        end
    elseif derivativeIDX==2
        if size(y)==(1,1)
            axpy!(1.0,sum(grad_c),grad_n)
        else
            tmp=CudaArray(Float64,size(grad_c))
            CUBLAS.geam!('T','N',1.0,grad_c,0.0,tmp,tmp) # tmp=grad_c'
            axpy!(1.0,tmp,grad_n); free(tmp)
        end
    end
end


end

Derivative[FXPYtranspose]=DXPYtranspose
Inplace[FXPYtranspose]=FXPYtranspose_inplace

export FXPYtranspose


