function getatisdata(maxlines)

    file=matopen("atis.mat")
    atis=read(file)
    seq=atis["seq"]

    if maxlines==0
       lines=length(seq)
    else
        lines=maxlines
    end

    MaxSeq=10000
    lexicon=Array(ASCIIString,572)
    tokens=Array(Any,lines)

    counter=0
    for l=1:lines
        if !any(seq[l].==0)
            counter+=1
            tokens[counter]=seq[l]
        end
    end
    tokens=tokens[1:counter]

    for i=1:length(atis["wd"])
        lexicon[i]=atis["wd"][i]
    end

return tokens,lexicon
end
