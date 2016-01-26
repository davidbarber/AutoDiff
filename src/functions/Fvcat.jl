# f(x)=vcat(x...)
Fvcat(x...)=(vcat(x...),nothing)
Fvcat_inplace(value::Array,auxvalue,x...)=copy!(value,vcat(x...))

function Dvcat(derivativeIDX,f_c,faux_c,grad_c,grad_n,x...)
    startind=1
    for i=1:length(x)
        endind=startind+length(x[i])-1
        if pointer(x[i])==pointer(x[derivativeIDX]) # use pointers so we can deal with vcat([A B A])
            axpy!(1.0,grad_c[startind:endind],grad_n)
        end
        startind=endind+1
    end
end


if PROC=="GPU"

    function Fvcat_inplace(value::CudaArray,auxvalue,x::CudaArray...) # inplace
        totallength=1
        for i in 1:length(x)
            copyinto!(value,x[i],totallength)
            totallength+=length(x[i])
        end
    end

    function Dvcat(derivativeIDX,f_c,faux_c,grad_c,grad_n::CudaArray,x...)
        startind=1
        for i=1:length(x)
            endind=startind+length(x[i])-1
            if pointer(x[i])==pointer(x[derivativeIDX]) # use pointers so we can deal with vcat([A B A])
                copyfrom_update!(grad_n,grad_c,startind)
            end
            startind=endind+1
        end
    end

end

Derivative[Fvcat]=Dvcat # Define dictionary lookup
Inplace[Fvcat]=Fvcat_inplace

import Base.vcat
vcat(n::ADnode)=ADFunction(Fvcat,n)

vcat(n::Array{ADnode,1})=n
vcat(n::ArrayADnode)=ADFunction(Fvcat,n)

export vcat
