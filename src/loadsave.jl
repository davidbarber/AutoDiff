#http://julia-programming-language.2336112.n4.nabble.com/Re-Any-method-to-save-the-variables-in-workspace-to-file-td3123.html

# load("filename")
function loadvars(filename)
  f = open(filename, "r")
  try
    eval(parse(readall(f)))
  finally
    close(f)
  end
end

#@savevars("filename",var)
macro savevars(filename, vars...)
  printexprs = map(vars) do var
    :(print(f, ";", $(string(var)), " = "); showall(f, $(esc(var))))
  end
  quote
    local f = open($(esc(filename)), "w")
    try
      $(Expr(:block, printexprs...))
    finally
      close(f)
    end
  end
end
