#---- KnetContainers ----
abstract KnetContainer <: KnetModule

function layers(c::KnetContainer)
   error("layers is not implemented for type ", typeof(c))
end


# The Sequential container
#=For instance
   cnn = Sequential(
      Conv4(3, 3, 3, 5),
      relu,
      pool,
      Conv4(3, 3, 5, 8),
      relu,
      x->pool(x;mode=2, window=size(x)[1:2]),
      Linear(...)
   )
=#
type Sequential <: KnetContainer
   layers::Array{Any, 1}
   Sequential(layers...) = new([l for l in layers])
end

function layers(s::Sequential)
   return s.layers
end

function forward(context, s::Sequential, x)
   o = x
   for l in s.layers
      if isa(l, KnetModule)
         o = @mc l(o)
      elseif isa(l, Function)
         o = l(o)
      else
         error("KnetContainer malformed: only modules and functions are allowed")
      end
   end
   return o
end

function add!(s::Sequential, m...)
   push!(s.layers, m...)
   return s
end


#= rescon = Table(+,
   Sequential(Conv4(...), BN(...), relu, Conv4(...), BN(...), relu),
   Shortcut(...)
)=#
type Table <: KnetContainer
   op::Function
   layers::Array{Any, 1}
   Table(op::Function, layers...) = new(op, [l for l in layers])
end

function layers(table::Table)
   return table.layers
end

function forward(context, t::Table, x)
   outputs = []
   for l in t.layers
      if isa(l, KnetContainer)
         push!(outputs, @mc l(x))
      elseif isa(l, Function)
         push!(outputs, l(x))
      else
         error("KnetContainer malformed: only functions and modules are allowed")
      end
   end
   t.op(outputs...)
end
