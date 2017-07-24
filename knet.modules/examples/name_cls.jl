using Knet, KnetModules
import KnetModules.forward

#=
Name Classification with GRU
data is taken from
http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#sphx-glr-intermediate-char-rnn-classification-tutorial-py
and downsampled to contain only Arabic, Chinese, Czech, Dutch,
Greek, Japanese and Scottish in order to balance the numbet of examples
(English was too big) and not dealing with unicode issues
=#
type Model <: KnetModule
   emb::Linear
   gru::GRU
   out::Linear
   auto_reset::Bool # automatically reset state at each forward
   Model(vsize, embsize, hsize, nclasses; auto_reset=true) = new(
      Linear(vsize, embsize),
      GRU(embsize, hsize),
      Linear(hsize, nclasses),
      auto_reset
   )
end

function forward(context, c::Model, seq, masks)
   if c.auto_reset
      reset_state!(c.gru)
   end
   for t = 1:length(seq)
      emb = @mc c.emb(seq[t])
      @mc c.gru(emb, masks[t])
   end
   @mc c.out(hidden_state(c.gru))
end

# Load the data
cd("names")
files = readdir()
x = []
y = []
class_names = []
dict = Dict{Any, Number}()
idict = 0
for (i, f) in enumerate(filter(f->endswith(f, ".txt"), files))
   push!(class_names, split(f, ".")[1])
   names = split(readstring(f), "\n")
   for n in names
      if length(n) < 2
         continue
      end
      push!(x, n)
      push!(y, i)
      for c in n
         if ~haskey(dict, c)
            idict += 1
            dict[c] = idict
         end
      end
   end
end
cd("..")

# shuffle tha data and divide the test split
inds = shuffle(1:length(x))
ntrn = Int(floor(.8length(inds)))
ntst = length(inds) - ntrn
println("Train/test split:", ntrn, "/", ntst)
xtrn = x[inds[1:ntrn]]
ytrn = y[inds[1:ntrn]]
xtst = x[inds[ntrn+1:end]]
ytst = y[inds[ntrn+1:end]]
# A minibatch of words
# where each row is a one-hot word
function mini_batch(dict, words, classes, nclasses; dtype=Array{Float32})
   seq = []
   masks = []
   max_len = maximum(map(length, words))
   dsize = length(keys(dict))
   for i = 1:max_len
      batch = dtype(zeros(dsize, length(words)))
      mask = [1 for _ in words]
      for j = 1:length(words)
         if length(words[j]) < i
            mask[j] = 0
         else
            try
               batch[dict[words[j][i]], j] = 1
            catch e
               println(words[j])
               error(e)
            end
         end
      end
      push!(seq, batch)
      push!(masks, dtype(mask))
   end
   labels = dtype(zeros(nclasses, length(words)))
   for i = 1:length(words)
      labels[classes[i], i] = 1
   end
   return seq, masks, labels
end

function report_accuracy(xtst, ytst;_print=false)
   ncorrect = 0
   for i = 1:length(xtst)
      s, m, _ = mini_batch(dict, xtst[i:i], ytst[i:i], length(class_names))
      if _print
         print(i, ": ", xtst[i])
      end
      scores = @run(model, s, m)
      ncorrect += Int(findmax(scores)[2] == ytst[i])
      if _print
         println("->", class_names[findmax(scores)[2]])
      end
      reset_state!(model.gru)
   end
   println("Accuracy: ", ncorrect/length(xtst))
end

# The main training loop
# Initialize the model and optimers
model = Model(length(dict), 64, 256, length(class_names))
model_params = parameters(model)
opts = [Adam() for p in model_params]
loss(y, ygold) = -sum(ygold .* logp(y)) ./ size(ygold, 2)
lossgrad = grad(model, loss)
for i = 1:100
   println("Epoch: ", i)
   print("Training ")
   report_accuracy(xtrn, ytrn)
   print("Test ")
   report_accuracy(xtst, ytst)
   for j = 1:32:length(xtrn)
      #println("j:", j)
      s, m, l = mini_batch(dict, xtrn[i:i+31], ytrn[i:i+31], length(class_names))
      g = lossgrad(s, m, l)
      for (o, p) in zip(opts, model_params)
         update!(get_value(p), get_grad(p, g), o)
      end
   end
   # Shuffle the data at each epoch
   inds = shuffle(1:ntrn)
   xtrn = xtrn[inds]
   ytrn = ytrn[inds]
end
report_accuracy(xtst, ytst; _print=true)
