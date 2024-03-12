# Activations and Gradients Statistics

> ***"... Really what I want to talk about is the importance of understanding the activations, the gradients and their statistics in neural networks. And this becomes increasingly important, especially as you make your networks bigger, larger and deeper."***
>
> *from Andrej Karpathy*

***Table of Contents***
<!-- no toc -->
- [*Basic MLP*](#basic-mlp)
- [*Initialization*](#initialization)
- [*Activations*](#activations)

## Basic MLP

First of all, let's rewrite the basic MLP more nicely.

```python
# dataset setup
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# read all the words in the file
words = open("../names.txt").read().splitlines()

# build the vocabulary of characters and mapping to/from integers
chars = sorted(list(set(''.join(words))))
stoi = { ch: i + 1 for i, ch in enumerate(chars) }
stoi["."] = 0
itos = { i: ch for ch, i in stoi.items() }
vocab_size = len(itos)

# build the dataset
block_size = 3  # using 3 contiguous characters to predict the next one
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]    # crop and append
    return torch.tensor(X), torch.tensor(Y)

# split the dataset, randomly
import random

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtrain  , Ytrain    = build_dataset(words[:n1])     # 80%
Xdev    , Ydev      = build_dataset(words[n1:n2])   # 10%
Xtest   , Ytest     = build_dataset(words[n2:])     # 10%
```

```python
# parameters setting
n_embd = 10     # the dimensionality of the character embedding vectors
n_hidden = 200  # the number of hidden units

g = torch.Generator().manual_seed(2147483647)   # consistent with Andrej's settings 
C = torch.randn((vocab_size, n_embd),                    generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden),   generator=g)
b1 = torch.randn((n_hidden,),                       generator=g)
W2 = torch.randn((n_hidden, vocab_size),            generator=g)
b2 = torch.randn((vocab_size,),                     generator=g)

parameters = [C, W1, b1, W2, b2]        # collect all parameters
print("#parameters in total:", sum(p.numel() for p in parameters))
for p in parameters:
    p.requires_grad = True
```

```text
#parameters in total: 11897
```

The training process is as follows:

```python
# model training
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construction
    ix = torch.randint(0, Xtrain.shape[0], (batch_size,))
    Xb, Yb = Xtrain[ix], Ytrain[ix]     # batch X, Y

    # forward pass
    emb = C[Xb]                                   # character embeddings
    embcat = emb.view(emb.shape[0], -1)           # concatenate the vectors
    hpreact = embcat @ W1 + b1                    # pre-activation
    h = torch.tanh(hpreact)                       # hidden layer
    logits = h @ W2 + b2                          # output layer
    loss = F.cross_entropy(logits, Yb)            # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01              # learning rate decay
    for p in parameters:
        p.data -= lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: loss={loss.item():.4f}")
    lossi.append(loss.item())
```

```text
      0/ 200000: loss=25.3606
  10000/ 200000: loss=2.8773
  20000/ 200000: loss=2.9392
  30000/ 200000: loss=2.2645
  40000/ 200000: loss=2.3070
  50000/ 200000: loss=2.1340
  60000/ 200000: loss=2.4111
  70000/ 200000: loss=2.1855
  80000/ 200000: loss=2.2445
  90000/ 200000: loss=2.1231
 100000/ 200000: loss=2.1515
 110000/ 200000: loss=1.9242
 120000/ 200000: loss=1.9465
 130000/ 200000: loss=2.0818
 140000/ 200000: loss=2.1904
 150000/ 200000: loss=1.9120
 160000/ 200000: loss=2.1915
 170000/ 200000: loss=2.0538
 180000/ 200000: loss=2.0643
 190000/ 200000: loss=2.3519
 ```

```python
# visualize the loss
plt.plot(lossi)
```

![loss-graph](../pictures/loss-graph.png)

```python
# model evaluation
@torch.no_grad()
def split_loss(split):
    x, y = {
        "train": (Xtrain, Ytrain),
        "val": (Xdev, Ydev),
        "test": (Xtest, Ytest)
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f"{split} loss: {loss.item():.4f}")

split_loss("train")
split_loss("val")
```

```text
train loss: 2.1244
val loss: 2.1692
```

```python
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):

    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor(context)]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        # sample
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        # shift and track
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    
    print(''.join(itos[i] for i in out))
```

```text
mona.
mayah.
seel.
nihayla.
rethan.
endraegelie.
koreliah.
milopalekenslen.
narleitzion.
kalin.
shubergiairiel.
kindreelynn.
nohalanu.
zayven.
kylani.
els.
kayshan.
kyla.
hil.
salynn.
```

So, this is the beginning of the journey.

## Initialization

For details, read [He et al. 2015](https://doi.org/10.48550/arXiv.1502.01852).

Now, we focus on the training process.

The first line of the output is strange. The loss is too large. How does it happen?

```text
0/ 200000: loss=25.3606
```

The first loss is `25.3606`, which is related to the initialization, directly. Let's deep dive into the initialization.

In the beginning, the distribution of the parameters is unknown, so a good guess is the probability of every char is the same. How much is the loss in this case?

```python
# if the probability of every char is the same
- torch.tensor(1/27.0).log().item() # the loss is 3.2958
```

`3.2958` is much smaller than `25.3606`. The problem is, after initialization, the probability of every char is messed up. Some chars are very confident, while others are very not confident. This is not rational. Andrej gives us a simpler and better way to test this idea.

```python
# 4-dimensional example of the issue

# if the probability of every char is the same
logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
probs = F.softmax(logits, dim=0)
loss = -probs[2].log()
print(f"probs: {probs}, loss: {loss.item():.4f}")
# probs: tensor([0.2500, 0.2500, 0.2500, 0.2500]), loss: 1.3863

# no matter what the logits are, the loss is the same
# the reason is the softmax compute structure
logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
probs = F.softmax(logits, dim=0)
loss = -probs[2].log()
print(f"probs: {probs}, loss: {loss.item():.4f}")
# probs: tensor([0.2500, 0.2500, 0.2500, 0.2500]), loss: 1.3863

# if the probabilities is messed up, and guess is right
logits = torch.tensor([0.0, 0.0, 5.0, 0.0])
probs = F.softmax(logits, dim=0)
loss = -probs[2].log()
print(f"probs: {probs}, loss: {loss.item():.4f}")
# probs: tensor([0.0066, 0.0066, 0.9802, 0.0066]), loss: 0.0200 (very low)

# if the probabilities is messed up, and guess is wrong
logits = torch.tensor([0.0, 0.0, 5.0, 0.0])
probs = F.softmax(logits, dim=0)
loss = -probs[1].log()
print(f"probs: {probs}, loss: {loss.item():.4f}")
# probs: tensor([0.0066, 0.0066, 0.9802, 0.0066]), loss: 5.0200 (very high)

# if generate the logits randomly following the gaussian
logits = torch.randn(4)
probs = F.softmax(logits, dim=0)
loss = -probs[2].log()
logits, probs, loss.item()
# (tensor([ 0.1984, -0.8986, -0.6969,  2.1464]),
#  tensor([0.1142, 0.0381, 0.0466, 0.8010]),
#  3.0651798248291016)

# if the logits are messed up, the loss is high
logits = torch.randn(4) * 10
probs = F.softmax(logits, dim=0)
loss = -probs[2].log()
logits, probs, loss.item()
# (tensor([  4.4975,  14.9176, -11.8179,  15.9216]),
#  tensor([7.9982e-06, 2.6815e-01, 6.5657e-13, 7.3184e-01]),
#  28.051748275756836)
```

Let's check the real `logits` in the model.

```python
# model training
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construction
    ix = torch.randint(0, Xtrain.shape[0], (batch_size,))
    Xb, Yb = Xtrain[ix], Ytrain[ix]     # batch X, Y

    # forward pass
    emb = C[Xb]                                   # character embeddings
    embcat = emb.view(emb.shape[0], -1)           # concatenate the vectors
    hpreact = embcat @ W1 + b1                    # pre-activation
    h = torch.tanh(hpreact)                       # hidden layer
    logits = h @ W2 + b2                          # output layer
    loss = F.cross_entropy(logits, Yb)            # loss function

    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: loss={loss.item():.4f}")
    lossi.append(loss.item())

    break   # set a break
```

```text
0/ 200000: loss=26.3527
```

```python
logits[0]
# tensor([ 12.4119,  -7.6996,   2.3776,   6.5568,  -6.7900, -15.6000, -21.2102,
#          -0.6893,  13.2828, -12.5420,  -4.2807,  25.9170,   1.7245, -19.7246,
#           2.6588,   7.7760, -15.6614,  14.8147,  16.6351,  -9.3979,  -6.0412,
#          -2.7174,  -1.9348,  -4.2945,  -9.4654,  -5.1644,   0.7409],
#        grad_fn=<SelectBackward0>)
# logits is messed up, so the loss is large
```

Based on the above, we want to initialize the parameters in a way that makes the probabilities of every char the same, roughly. `logits` are calculated by `h @ W2 + b2`, so if we make the `W2` and `b2` close to zero, the `logits` will be close to zero, and the `probs` will be close to `1/vocab_size`. This is how we do.

```python
C = torch.randn((vocab_size, n_embd),               generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden),   generator=g)
b1 = torch.randn((n_hidden,),                       generator=g)
W2 = torch.randn((n_hidden, vocab_size),            generator=g) * 0.01 # close to zero
b2 = torch.randn((vocab_size,),                     generator=g) * 0.0  # set to zero
```

and then, we re-run the code as above. The first loss is much smaller.

```python
# 0/ 200000: loss=3.3111
```

As expected, the first loss is close to 3.2958.

```python
logits[0]

#  tensor([ 0.1062, -0.1247, -0.0880,  0.0447,  0.0015,  0.0700,  0.1914,  0.0085,
#          0.1691,  0.0144, -0.0410, -0.0284, -0.0890,  0.0562, -0.0406,  0.1880,
#         -0.1316, -0.0452,  0.1353,  0.0273,  0.0219,  0.0505,  0.0950, -0.1829,
#          0.0873, -0.0365, -0.1384], grad_fn=<SelectBackward0>)
```

After re-running the training, the final result is better than before, and the hockey-shaped appearance disappears. So our training is much more efficient. When initialing better, training doesn't waste time on fixing the ridiculous guesses.

```text
train loss: 2.0704
val loss: 2.1345
```

![loss-better-init.png](../pictures/loss-better-init.png)

## Activations

The problem now is with the values of `h`, the activations of the hidden states.

```python
h

# tensor([[ 0.9970, -1.0000, -1.0000,  ..., -0.9989, -0.9997,  0.9432],
#         [-0.9941, -0.9997, -1.0000,  ..., -0.9900,  0.9994,  1.0000],
#         [-1.0000,  0.8103, -0.9994,  ...,  0.9998,  1.0000,  0.9988],
#         ...,
#         [-1.0000, -0.9957, -0.9435,  ...,  1.0000, -1.0000, -0.9999],
#         [-0.9793, -0.9999, -1.0000,  ...,  0.7836, -0.7058,  0.2913],
#         [ 1.0000, -0.5961,  1.0000,  ..., -0.7709, -0.9997,  0.9992]],
#        grad_fn=<TanhBackward0>)
```

There are a lot of `1.0` and `-1.0` in the `h`. This is not good. The `tanh` function is supposed to squash the values to the range `[-1, 1]`, but it's not doing a good job.

![h-hist](../pictures/h-hist.png)

In the implementation, when `t` is close to `1` or `-1`, the `backward` of `tanh` is close to `0`, which means the gradient vanishes.

```python
def tanh(x):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t ** 2) * out.grad    # t = 1 or -1 --> grad = 0 vanishes!
    out._backward = _backward
    
    return out
```

Let's see how many `1.0` and `-1.0` are in the `h`, roughly.

```python
plt.figure(figsize=(15, 5))
plt.imshow(h.abs() > 0.99, cmap="gray", interpolation='nearest')
```

![Hot Graph](../pictures/hot-graph.png)

How to fix this? The key is to re-range `hpreact`, which squashes the values closer to zero, before applying `tanh`. To achieve this, we can initialize `W1` and `b1` in a way that makes the pre-activations close to zero.

```python
W1 = torch.randn((n_embd * block_size, n_hidden),   generator=g) * 0.2
b1 = torch.randn((n_hidden,),                       generator=g) * 0.01
```

After re-running the training, the activations are much better.

```python
plt.hist(h.view(-1).tolist(), bins=50)
```

![h-hist-better](../pictures/h-hist-better.png)

And the `tanh` squashes the values to the range `[-1, 1]` as expected.

```python
plt.figure(figsize=(15, 5))
plt.imshow(h.abs() > 0.99, cmap="gray", interpolation='nearest')
```

![Hot Graph Better](../pictures/hot-graph-better.png)

Everything is fine now. Again, after re-running the whole code, we get a better result.

```text
train loss: 2.0370
val loss: 2.1059
```

> ***"... And the deeper your network is and the more complex it is, the less forgiving it is to these errors. ( from initialization )"***
>
> *from Andrej Karpathy*

Now let's find out the formula to initialize `W1` and `b1` instead of setting manually. Again, Andrej gives us a piece of great code to motivate the discussion of this.

```python
# torch.randn generates the values following the gaussian (0, 1)
x = torch.randn(1000, 10)
w = torch.randn(10, 200)
y = x @ w
print(x.mean(), x.std())
print(y.mean(), y.std())
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.hist(x.view(-1).tolist(), bins=50, density=True);
plt.subplot(1, 2, 2)
plt.hist(y.view(-1).tolist(), bins=50, density=True);
```

```text
tensor(0.0011) tensor(0.9886)
tensor(0.0055) tensor(3.1956)   # the multiplication makes dist more spread
```

![x-y-hist](../pictures/x-y-hist.png)

The multiplication makes distribution more spread. And so the question is, how do we scale these `w` to preserve this distribution to remain a Gaussian?

