function genFcode(v,av,n::Array{ADnode,1},fname=[])
# generate source code for the Autodiff forward pass
# (c) David Barber, University College London 2015    
   if isempty(fname)
       iostream=STDOUT
       str="function ("
   else
       iostream=open(fname*"!.jl","w")
       str="function "*fname*"!("
   end
    N=length(n)
    str=str*"v,av)"
    println(iostream,str)

    # forward pass:
    for i=1:N
        if !(n[i].input)
            str="v[$i],av[$i]="*string(n[i].f)*"("
            for j=n[i].parents
                tstr="::"*string(typeof(v[j]))
                str=str*"v[$j]"*tstr*","
            end
            str=str[1:end-1]*")"
            println(iostream,str)
        end
    end
    println(iostream,"end")
    if !isempty(fname)
        close(iostream)
    end
end



