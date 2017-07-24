#---Recurrent modules---

abstract AbstractRNN <: KnetModule

function reset_state!(rnn::AbstractRNN)
   rnn.history = []
end

function hidden_state(rnn::AbstractRNN; unbox=false)
   unbox ? AutoGrad.getval(rnn.history[end]) : rnn.history[end]
end

# The GRU layer
type GRU <: AbstractRNN
   history::Array{Any, 1}
   h2z::Linear
   i2z::Linear
   h2r::Linear
   i2r::Linear
   h2n::Linear
   i2n::Linear
   input_size::Int
   hidden_size::Int
   function GRU(input_size::Int, hidden_size::Int; bias=true)
      h2z = Linear(hidden_size, hidden_size; bias=bias)
      i2z = Linear(input_size, hidden_size; bias=bias)
      h2r = Linear(hidden_size, hidden_size; bias=bias)
      i2r = Linear(input_size, hidden_size; bias=bias)
      h2n = Linear(hidden_size, hidden_size; bias=bias)
      i2n = Linear(input_size, hidden_size; bias=bias)
      new([], h2z, i2z, h2r, i2r, h2n, i2n, input_size, hidden_size)
   end
end

# Single time step of a GRU
# TODO: checkout masking
function forward(context, m::GRU, x, mask=nothing)
   if length(m.history) == 0
      tx = typeof(AutoGrad.getval(x))
      h0 = zeros(m.hidden_size, size(x)[end])
      push!(m.history, tx(h0))
   end
   ht_1 = hidden_state(m)
   z = sigm(@mc(m.i2z(x)) .+ @mc(m.h2z(ht_1)))
   r = sigm(@mc(m.i2r(x)) .+ @mc(m.h2r(ht_1)))
   n = tanh(@mc(m.i2n(x)) .+ r .* @mc(m.h2n(ht_1)))
   if mask != nothing && sum(mask) < length(mask)
      if ndims(mask) == 1
         mask = reshape(mask, (1, size(mask)[1]))
      end
      assert(size(mask) == (1, size(x)[2]))
      #= The idea is destroying the effect of padded values
      from the output and freeze their state
      but I'm not sure. =#
      z = z .* mask
      r = r .* mask
      n = n .* mask
      ht = (1 .- z) .* n .+ z .* ht_1 .+ (1 .- mask) .* ht_1
   else
      ht = (1 .- z) .* n .+ z .* ht_1
   end
   push!(m.history, ht)[end]
end
