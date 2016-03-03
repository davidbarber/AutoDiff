# f(x)=hcat(x...)
# need to think a little bit
Fhcat(x...)=(hcat(x...),nothing)
Fhcat_inplace(handle,value,auxvalue,x...)=copy!(value,hcat(x...))

function Dhcat(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n,x...)
    startind=1
    for i=1:length(x)
        endind=startind+length(x[i])-1
        if pointer(x[i])==pointer(x[derivativeIDX]) # use pointers so we can deal with hcat([A B A])
            axpy!(1.0,grad_c[startind:endind],grad_n)
        end
        startind=endind+1
    end
end


if PROC=="GPU"

    function Fhcat_inplace(handle,value::CudaArray,auxvalue,x::CudaArray...) # inplace
        totallength=1
        for i in 1:length(x)
            copyinto!(value,x[i],totallength)
            totallength+=length(x[i])
        end
    end

    function Dhcat(handle,derivativeIDX,f_c,faux_c,grad_c,grad_n::CudaArray,x...)
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

Derivative[Fhcat]=Dhcat # Define dictionary lookup
Inplace[Fhcat]=Fhcat_inplace

import Base.hcat
hcat(n::ADnode)=ADFunction(Fhcat,n)

hcat(n::Array{ADnode,1})=n
hcat(n::ArrayADnode)=ADFunction(Fhcat,n)

export hcat
