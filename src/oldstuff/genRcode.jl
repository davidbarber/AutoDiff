function genRcode(v,av,g,n::Array{ADnode,1},mess,fname=[])
# generate source code for reverse Autodiff pass
# (c) David Barber, University College London 2015    
   if isempty(fname)
       iostream=STDOUT
       str="function ("
   else
       iostream=open(fname*"!.jl","w")
       str="function "*fname*"!("
   end
    N=length(n)

    str=str*"v,av,g)"
    println(iostream,str)

    # reverse pass:
    reverseorder=N:-1:1
    println(iostream,"g[$N]=1.0")
    for i in reverseorder[2:N]
        if any(n[i].takederivative)
            count=0
            for c in n[i].children
                str="g[$i]="
                count+=1
                if count==1
                    preval=""
                else
                    preval="g[$i]+"
                end
                str=str*preval*string(mess[i,c])*"("
                for j in n[c].parents
                    tstr="::"*string(typeof(v[j]))
                    str=str*"v[$j]"*tstr*","
                end
                tstr="::"*string(typeof(v[c]))
                str=str*"v[$c]"*tstr*","
                tstr="::"*string(typeof(av[c]))
                str=str*"av[$c]"*tstr*","
                tstr="::"*string(typeof(g[c]))
                str=str*"g[$c]"*tstr*")"
                println(iostream,str)
            end
        end
    end
    println(iostream,"end")
    if !isempty(fname)
        close(iostream)
    end
end



