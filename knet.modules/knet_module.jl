# KnetModule abstraction
abstract KnetModule

function gpu!(m::KnetModule; o...)
   if Knet.gpu() < 0
      warn("No available gpu, transfer ignored, using cpu...")
      return
   end
   for p in parameters(m)
      gpu!(p; o...)
   end
end

function cpu!(m::KnetModule; o...)
   for p in parameters(m)
      cpu!(p; o...)
   end
end

#Macro for calling types directly
macro mc(expr)
   if typeof(expr) == Expr && expr.head == :call
      return esc(:(forward(context, $(expr.args[1]), $(expr.args[2:end]...))))
      #return Expr(:call, :forward, :context, expr.args[1], expr.args[2:end]...)
   end
   return expr
end

# Macro for running modules in active context
macro run(m, args...)
   esc(:(forward(get_active_parameter_context(), $m, $(args...))))
end

function _populate_recursive(m, list, match_type)
   if isa(m, match_type)
      push!(list, m)
   end

   if isa(m, KnetModule)
      for fn in fieldnames(m)
         _populate_recursive(getfield(m, fn), list, match_type)
      end
   elseif isa(m, Array) || isa(m, Tuple)
      for l in m
         _populate_recursive(l, list, match_type)
      end
   elseif isa(m, Associative)
      for v in values(m)
         _populate_recursive(v, list, match_type)
      end
   else
      return
   end
end

function parameters(m::KnetModule)
   res = []
   _populate_recursive(m, res, Parameter)
   return res
end

# Contains itself
function submodules(m::KnetModule)
   res = []
   _populate_recursive(m, res, KnetModule)
   return res
end


function set_mode!(m::KnetModule, mode)
   for sm in submodules(m)
      if :mode in fieldnames(sm)
         sm.mode = mode
      end
   end
end

training!(m::KnetModule) = set_mode!(m, :train)

testing!(m::KnetModule) = set_mode!(m, :test)

import AutoGrad.grad
function grad(m::KnetModule, loss)
   _predict(w, x...) = forward(w, m, x...)
   _loss(w, args...) = loss(_predict(w, args[1:end-1]...), args[end])
   lossgrad = grad(_loss)
   return (args...)->lossgrad(get_active_parameter_context(), args[1:end-1]..., args[end])
end
