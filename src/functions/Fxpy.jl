# f(x,y)=x+y

function Fxpy(x::Float64,y::Float64)
    return ((x+y)*ones(1,1),nothing)
end

function Fxpy(x::Float64,y)
    return (x*ones(size(y))+y,nothing)
end

function Fxpy(x,y::Float64)
    return (y*ones(size(y))+x,nothing)
end

function Fxpy(x,y)
    if size(x)==(1,1)
        return (x[1]*ones(size(y))+y,nothing)
    elseif size(y)==(1,1)
        return (x+y[1]*ones(size(x)),nothing)
    else
        return (x.+y,nothing)
    end
end

function Fxpy_inplace(value,auxvalue,x,y)
    if size(x)==(1,1)
        copy!(value,x[1]*ones(size(y))); axpy!(1.0,y,value)
    elseif size(y)==(1,1)
        copy!(value,y[1]*ones(size(x))); axpy!(1.0,x,value)
    else
        copy!(value,x); axpy!(1.0,y,value)
    end
end

function Dxpy(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::Array,y::Array)
    if size(x)==(1,1) && derivativeIDX==1
        axpy!(1.0,[sum(grad_c)],grad_n)
    elseif size(y)==(1,1) && derivativeIDX==2
        axpy!(1.0,[sum(grad_c)],grad_n)
    else
        axpy!(1.0,grad_c,grad_n)
    end
end

if PROC=="GPU"

    function Fxpy(x::CudaArray,y::CudaArray)
        if size(x)==(1,1)
            tmp=CudaArray(Float64,size(y))
            gfill!(tmp,x)
            axpy!(1.0,y,tmp)
        elseif size(y)==(1,1)
            tmp=CudaArray(Float64,size(x))
            gfill!(tmp,y)
            axpy!(1.0,x,tmp)
        else
            tmp=CudaArray(zeros(size(x))); copy!(tmp,x); axpy!(1.0,y,tmp)
        end
        return (tmp,nothing)
    end

    function Fxpy_inplace(value::CudaArray,auxvalue,x::CudaArray,y::CudaArray)
        if size(x)==(1,1)
            gfill!(value,x); axpy!(1.0,y,value)
        elseif size(y)==(1,1)
            gfill!(value,y); axpy!(1.0,x,value)
        else
            copy!(value,x); axpy!(1.0,y,value)
        end
    end

    function Dxpy(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray,y::CudaArray)

        if size(x)==(1,1) && derivativeIDX==1
            axpy!(1.0,sum(grad_c),grad_n)
        elseif size(y)==(1,1) && derivativeIDX==2
            axpy!(1.0,sum(grad_c),grad_n)
        else
            axpy!(1.0,grad_c,grad_n)
        end
    end
end

Derivative[Fxpy]=Dxpy
Inplace[Fxpy]=Fxpy_inplace

import Base.+
+(A::ADnode,B::ADnode)=ADnode(Fxpy,[A B])
+(A::Real,B::ADnode)=ADnode(Fxpy,[ADconst(A) B])
+(A::ADnode,B::Real)=ADnode(Fxpy,[A ADconst(B)])



if PROC=="GPU"
    function +(A::CudaArray,B::CudaArray)
#        tmp=CudaArray(zeros(size(A))); copy!(tmp,A); axpy!(1.0,B,tmp);
#        return tmp
        return CUBLAS.geam('N','N',1.0,A,1.0,B)
    end

    function +(A::CudaArray,B::CudaArray,out::CudaArray)
#        fill!(out,0.0); axpy!(1.0,A,out);axpy!(1.0,B,out)
        CUBLAS.geam!('N','N',1.0,A,1.0,B,out)
    end

end

export Fxpy, +


