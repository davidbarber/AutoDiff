# mean square loss: f(x,y)=(sum_{i=1:I} (x_i-y_i)^2)/I
# One way to do this is meansquare( x-y). However, this would define a new now x-y. This would be fast, but require storing x-y.
# As an alternative, we'll recompute here x-y, rather than storing it. This will be slower, but saves on storage.
function FmeanSquareLoss(malloc::Bool,x,y)
return (1,1)
end


FmeanSquareLoss(x,y)=(FmeanSquare(axpy(-1.0,y,x))[1],nothing)
function FmeanSquareLoss_inplace(value,auxvalue,x,y)
        FmeanSquare_inplace(value,auxvalue,axpy(-1.0,y,x))
end


function DmeanSquareLoss(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::Array,y::Array)
    if derivativeIDX==1
        tmp=copy(grad_c)  # how to avoid this tmp?
        scale!(2./length(x),tmp)
        axpy!(tmp[1],axpy(-1.0,y,x),grad_n)
    elseif derivativeIDX==2
        tmp=copy(grad_c)
        scale!(2./length(x),tmp)
        axpy!(tmp[1],axpy(-1.0,x,y),grad_n)
    end
end

if PROC=="GPU"

    function FmeanSquareLoss(x::CudaArray,y::CudaArray)
        tmp=copy(x)
        axpy!(-1.0,y,tmp)
        return (FmeanSquare(tmp)[1],nothing)
    end

    function FmeanSquareLoss_inplace(handle,value,auxvalue,x::CudaArray,y::CudaArray)
        #copy!(value,x)
        #axpy!(-1.0,y,value);FmeanSquare_inplace(value,auxvalue,value)
        # why doesn't this work??

        tmp=copy(x)
        axpy!(-1.0,y,tmp);FmeanSquare_inplace(value,auxvalue,tmp)
        free(tmp)
    end


    function DmeanSquareLoss(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)
        if derivativeIDX==1
            DmeanSquareLoss!(grad_c,x,y,grad_n)
        elseif derivativeIDX==2   
            DmeanSquareLoss!(grad_c,y,x,grad_n)
        end
    end
end

Derivative[FmeanSquareLoss]=DmeanSquareLoss
Inplace[FmeanSquareLoss]=FmeanSquareLoss_inplace

meanSquareLoss(nx::ADnode,ny::ADnode)=ADFunction(FmeanSquareLoss,nx,ny)
export  meanSquareLoss
