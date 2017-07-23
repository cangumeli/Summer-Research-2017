# Convolution variants
# will go here

type Conv4 <: KnetModule
   cfg
   weight
   bias
   Conv4(height::Int, width::Int, input::Int, output::Int;
      bias=true, dtype=Array{Float32}, o...) = begin
         w = Parameter(dtype(randn(height, width, input, output) *
               sqrt(2.0 / (height * width * output))))
         b = bias ? Parameter(dtype(zeros(1, 1, output, 1))) : nothing
         new(o, w, b)
   end
end

function forward(context, c::Conv4, x)
   o = @po conv4(c.weight, x; mode=1, c.cfg...)
   if c.bias !== nothing
      o = @po o .+ c.bias
   end
   return o
end
