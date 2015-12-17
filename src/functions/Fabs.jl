# f(x)=abs(x)

function Fabs(x)
    return (abs(x),nothing)
end

function Fabs_inplace(value::Array,auxvalue,x) # inplace
    copy!(value,abs(x))
end


function Dabs(derivativeIDX,f_c,faux_c,grad_c,grad_n,x)
    axpy!(1.0,sign(x).*grad_c,grad_n)
end

if 1==0 # TODO

    function uuabs(A::CudaArray)
        return CudaArray(CUBLAS.asum(A)/length(A)*ones(1,1))
    end
    export meanAbs

    function meanAbs!(A::CudaArray,Out::CudaArray)
        copy!(Out,meanAbs(A))
    end
    export meanAbs!


    function FmeanAbs(x::CudaArray...)
        tmp=CudaArray(zeros(1,1))
        for i in 1:length(x)
            axpy!(1.0/length(x),meanAbs(x[i]),tmp)
        end
        return (tmp,nothing)
    end


    function FmeanAbs_inplace(value::CudaArray,auxvalue,x::CudaArray...) # inplace
        fill!(value,0.0)
        for i in 1:length(x)
            axpy!(1.0/length(x),meanAbs(x[i]),value)
        end
    end


    function DmeanAbs(derivativeIDX,f_c,faux_c,grad_c,grad_n,x::CudaArray...)
        tmp=CudaArray(Float64,size(grad_n))
        vsign!(x[derivativeIDX],tmp)
        alphaaxpy!(1.0/(length(x)*length(x[derivativeIDX])),grad_c,tmp,grad_n)
        free(tmp)
    end

end

Derivative[Fabs]=Dabs # Define dictionary lookup
Inplace[Fabs]=Fabs_inplace

import Base.abs

abs(n::ADnode)=ADnode(Fabs,n)

###abs(A::ADtrans)=transpose(abs(node[A.parent])) # abs(A')=(abs(A))' TODO:check

export abs
