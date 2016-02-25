Flog(x)=(log(x),nothing)

function Flog_inplace(value,auxvalue,x)
    copy!(value,log(x))
end

function Dlog(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)
    axpy!(1.0,grad_c./f_c,grad_n)
end


if PROC=="GPU"

    function Flog(x::CudaArray)
        tmp=CudaArray(Float64,size(x))
        log!(x,tmp)
        return (tmp,nothing) # memory leak here
    end
    
    function Flog_inplace(handle,value,auxvalue,x::CudaArray)
        log!(x,value)
    end
    
    function Dlog(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray)
        vdivupdate!(1.0,grad_c,f_c,grad_n)
    end

end


Derivative[Flog]=Dlog
Inplace[Flog]=Flog_inplace

import Base.log

log(n::ADnode)=ADFunction(Flog,n)

export log
