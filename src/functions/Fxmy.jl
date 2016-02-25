# f(x,y)=x-y


function Fxmy(x::Float64,y::Float64)
    return ((x-y)*ones(1,1),nothing)
end

function Fxmy(x::Float64,y)
    return (x*ones(size(y))-y,nothing)
end

function Fxmy(x,y::Float64)
    return (x-y*ones(size(y)),nothing)
end


function Fxmy(x,y)
    if size(x)==(1,1)
        return (x[1]*ones(size(y))-y,nothing)
    elseif size(y)==(1,1)
        return (x-y[1]*ones(size(x)),nothing)
    else
        return (x.-y,nothing)
    end
end

function Fxmy_inplace(handle,value,auxvalue,x,y)
    if size(x)==(1,1)
        copy!(value,x[1]*ones(size(y))); axpy!(-1.0,y,value)
    elseif size(y)==(1,1)
        copy!(value,x); axpy!(-1.0,y[1]*ones(size(x)),value)
    else
        copy!(value,x); axpy!(-1.0,y,value)
    end
end

function Dxmy(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::Array,y::Array)
    if derivativeIDX==1
        if size(x)==(1,1)
            axpy!(1.0,[sum(grad_c)],grad_n)
        else
            axpy!(1.0,grad_c,grad_n)
        end
    elseif derivativeIDX==2
        if size(y)==(1,1)
            axpy!(-1.0,[sum(grad_c)],grad_n)
        else
            axpy!(-1.0,grad_c,grad_n)
        end
    end
end

if PROC=="GPU"
function Fxmy(x::CudaArray,y::CudaArray)
        if size(x)==(1,1)
            tmp=CudaArray(Float64,size(y))
            gfill!(tmp,x)
            axpy!(-1.0,y,tmp)
        elseif size(y)==(1,1)
            tmp=CudaArray(Float64,size(x))
            gfill!(tmp,y); scale!(-1.0,tmp)
            axpy!(1.0,x,tmp)
        else
            tmp=CudaArray(zeros(size(x))); copy!(tmp,x); axpy!(1.0,y,tmp)
        end
        return (tmp,nothing)
    end

    function Fxmy_inplace(handle,value::CudaArray,auxvalue,x::CudaArray,y::CudaArray)
        if size(x)==(1,1)
            gfill!(value,x); axpy!(-1.0,y,value)
        elseif size(y)==(1,1)
            gfill!(value,y); scale!(-1.0,value); axpy!(1.0,x,value)
        else
            copy!(value,x); axpy!(1.0,y,value)
        end
    end

    function Dxmy(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)

        if derivativeIDX==1
            if size(x)==(1,1)
                axpy!(1.0,sum(grad_c),grad_n)
            else
                axpy!(1.0,grad_c,grad_n)
            end
        elseif derivativeIDX==2
            if size(y)==(1,1)
                axpy!(-1.0,sum(grad_c),grad_n)
            else
                axpy!(-1.0,grad_c,grad_n)
            end
        end
    end

end

Derivative[Fxmy]=Dxmy
Inplace[Fxmy]=Fxmy_inplace

import Base.-
#=
-(A::ADnode,B::ADnode)=ADFunction(Fxmy,[A B])
-(A::Real,B::ADnode)=ADFunction(Fxmy,[ADconst(A) B])
-(A::ADnode,B::Real)=ADFunction(Fxmy,[A ADconst(B)])
=#
-(A::ADnode,B::ADnode)=ADFunction(Fxmy,A,B)
-(A::Real,B::ADnode)=ADFunction(Fxmy,ADconst(Float64(A)),B)
-(A::ADnode,B::Real)=ADFunction(Fxmy,A,ADconst(Float64(B)))


export Fxmy, -


