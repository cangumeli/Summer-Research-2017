# Linear and unary layers
type Linear <: KnetModule
   weight
   bias
   Linear(input::Int, output::Int; dtype=Array{Float32}, bias=true) = begin
      stdv = 1 ./ sqrt(input)
      uniform(dims) = rand(dims) * 2stdv - stdv
      w = Parameter(dtype(uniform((output, input))))
      b = bias ? Parameter(dtype(zeros((output, 1)))) : nothing
      new(w, b)
   end
end

function forward(context, l::Linear, x)
   if ndims(x) > 2
      x = mat(x)
   end
   o = @po l.weight * x
   if l.bias !== nothing
      o = @po o .+ l.bias
   end
   return o
end

type Dropout <: KnetModule
   pdrop::AbstractFloat
   mode::Symbol
   noise
   Dropout(pdrop::AbstractFloat; mode=:train) = new(pdrop, mode, nothing)
end

function forward(context, m::Dropout, x)
   assert(m.mode in [:train, :test])
   tx = typeof(AutoGrad.getval(x))
   sizex = size(x)
   p = eltype(x)(m.pdrop)
   if m.mode == :train
      dropout(x, p)
   else
      (eltype(x)(1) - p) .* x
   end
end
