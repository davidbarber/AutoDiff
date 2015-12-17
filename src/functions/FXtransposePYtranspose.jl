# f(x,y)=x'+y'

FXtransposePYtranspose(x::Float64,y::Float64)=Fxpy(x,y)
FXtransposePYtranspose(x::Float64,y)=Fxpy(x,y')

FXtransposePYtranspose(x,y::Float64)=Fxpy(x,y')

FXtransposePYtranspose(x::Array,y::Array)=Fxpy(x',y')

FXtransposePYtranspose_inplace(value,auxvalue,x::Array,y::Array)=Fxpy_inplace(value,auxvalue,x',y')

function DXtransposePYtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::Array,y::Array)
    if size(x)==(1,1) && derivativeIDX==1
        axpy!(1.0,[sum(grad_c)],grad_n)
    elseif size(y)==(1,1) && derivativeIDX==2
        axpy!(1.0,[sum(grad_c)],grad_n)
    else
        axpy!(1.0,grad_c',grad_n)
    end
end


if GPU
    function FXtransposePYtranspose_inplace(value,auxvalue,x::CudaArray,y::CudaArray)
        if size(x)==(1,1)
            FXPYtranspose_inplace(value,auxvalue,x::CudaArray,y::CudaArray)
        elseif size(y)==(1,1)
            FXtransposePY_inplace(value,auxvalue,x::CudaArray,y::CudaArray)
        else
            CUBLAS.geam!('T','T',1.0,x,1.0,y,value)
        end
    end

    function DXtransposePYtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        if derivativeIDX==1
            DXtransposePY(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        elseif derivativeIDX==2
            DXPYtranspose(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        end
    end

end


Derivative[FXtransposePYtranspose]=DXtransposePYtranspose
Inplace[FXtransposePYtranspose]=FXtransposePYtranspose_inplace

export FXtransposePYtranspose

