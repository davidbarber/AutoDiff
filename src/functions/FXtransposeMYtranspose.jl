# f(x,y)=x'-y'

FXtransposeMYtranspose(x::Float64,y::Float64)=Fxmy(x,y)
FXtransposeMYtranspose(x::Float64,y)=Fxmy(x,y')
FXtransposeMYtranspose(x,y::Float64)=Fxmy(x',y)
FXtransposeMYtranspose(x,y)=Fxmy(x',y')
FXtransposeMYtranspose_inplace(handle,value,auxvalue,x,y)=Fxmy_inplace(value,auxvalue,x',y')

function DXtransposeMYtranspose(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::Array,y::Array)
    if derivativeIDX==1
        if size(x)==(1,1)
            axpy!(1.0,[sum(grad_c)],grad_n)
        else
            axpy!(1.0,grad_c',grad_n)
        end
    elseif derivativeIDX==2
        if size(y)==(1,1)
            axpy!(-1.0,[sum(grad_c)],grad_n)
        else
            axpy!(-1.0,grad_c',grad_n)
        end
    end
end



if PROC=="GPU"

    function FXtransposeMYtranspose_inplace(handle,value,auxvalue,x::CudaArray,y::CudaArray)
        if size(x)==(1,1)
            FXMYtranspose_inplace(value,auxvalue,x::CudaArray,y::CudaArray)
        elseif size(y)==(1,1)
            FXtransposeMY_inplace(value,auxvalue,x::CudaArray,y::CudaArray)
        else
            CUBLAS.geam!('T','T',1.0,x,-1.0,y,value)
        end
    end

    function DXtransposeMYtranspose(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        if derivativeIDX==1
            DXtransposeMY(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        elseif derivativeIDX==2
            DXMYtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        end
    end

end

Derivative[FXtransposeMYtranspose]=DXtransposeMYtranspose
Inplace[FXtransposeMYtranspose]=FXtransposeMYtranspose_inplace


export FXtransposeMYtranspose


