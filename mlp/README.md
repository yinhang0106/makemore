# Multilayer Perceptron (MLP)

![MLP](../pictures/MLP.png)

MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

***Table of Contents***
<!-- no toc -->
- [*Usage*](#usage)
- [*Basic Setup*](#basic-setup)
- [*Build MLP*](#build-mlp)
- [*Training*](#training)
- [*Mini-Batch*](#mini-batch)
- [*Learning Rate*](#learning-rate)

## Usage

The included `names.txt` dataset, as an example, has the most common 32K names taken from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```text
emma
olivia
ava
isabella
sophia
charlotte
...
```

## Basic Setup

First, we build the dataset to train.

```python
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

# build the dataset
block_size = 3  # context length: how many characters do we take to predict the next one
X, Y = [], []
for w in words:
    
    print(w)
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join([itos[i] for i in context]), '--->', itos[ix])
        context = context[1:] + [ix]    # crop and append
        
X = torch.tensor(X)
Y = torch.tensor(Y)
```

How many samples do we have?

```python
len(words)  # 32033
```

Let's see how we transformed the dataset to features and labels.

```python
block_size = 3  # using contiguous 3 characters to predict the next one
X, Y = [], []
for w in words[:3]:
    print(w)
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join([itos[i] for i in context]), '--->', itos[ix])
        context = context[1:] + [ix]    # crop and append
```

```text
emma
... ---> e
..e ---> m
.em ---> m
emm ---> a
mma ---> .
olivia
... ---> o
..o ---> l
.ol ---> i
oli ---> v
liv ---> i
ivi ---> a
via ---> .
ava
... ---> a
..a ---> v
.av ---> a
ava ---> .
```

## Embedding

In this case, we want to embed the features into a 2D space. So, let's do it.

First, generate the embedding matrix.

```python
g = torch.Generator().manual_seed(2147483647)   # consistent with Andrej's settings
C = torch.randn((27, 2), generator=g)
```

Why $27 \times 2$? Because we map 27 characters ('a...z' + '.') to a 2D space.

simply run blow, we get a 3D tensor.

```python
emb = C[X]
emb.shape   # torch.Size([228146, 3, 2])
```

How `emb = C[X]` work? It's a kind of indexing. When indexing `C` with `X`, we get a 3D tensor. **The first dimension is the same as `X`, and the last two dimensions are the same as `C`.**

For example. The embedding matrix `C` is below:

```text
# C = torch.randn((27, 2), generator=g)
tensor([[ 1.5674, -0.2373],
        [-0.0274, -1.1008],
        [ 0.2859, -0.0296],
        [-1.5471,  0.6049],
        [ 0.0791,  0.9046],
        [-0.4713,  0.7868],
        [-0.3284, -0.4330],
        [ 1.3729,  2.9334],
        [ 1.5618, -1.6261],
        [ 0.6772, -0.8404],
        [ 0.9849, -0.1484],
        [-1.4795,  0.4483],
        [-0.0707,  2.4968],
        [ 2.4448, -0.6701],
        [-1.2199,  0.3031],
        [-1.0725,  0.7276],
        [ 0.0511,  1.3095],
        [-0.8022, -0.8504],
        [-1.8068,  1.2523],
        [ 0.1476, -1.0006],
        [-0.5030, -1.0660],
        [ 0.8480,  2.0275],
        [-0.1158, -1.2078],
        [-1.0406, -1.5367],
        [-0.5132,  0.2961],
        [-1.4904, -0.2838],
        [ 0.2569,  0.2130]])
```

Let's embed "..e" manually:

1. `..e` is `[0, 0, 5]` in `X` depending on the mapping `stoi`.
2. `[0, 0, 5]` to index means `['the first row', 'the first row', 'the 6th row']` in `C`, so we get `[[ 1.5674, -0.2373], [ 1.5674, -0.2373], [ 0.4713,  0.7868]]`.

Now check the result:

```python
emb[1]  # "..e" is the 2nd sample in X
```

```text
tensor([[ 1.5674, -0.2373],
        [ 1.5674, -0.2373],
        [-0.4713,  0.7868]])    # exactly the same as we calculated.
```

Next, we want to flatten the 3D tensor to 2D, so we can use it as input to the MLP. Mapping `(228146, 3, 2)` to `(228146, 6)`. And Andrej gives us three ways to implement it.

```python
# Method 1. Using torch.cat
torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], dim=1)

# Method 2. Using torch.cat and torch.unbind
torch.cat(torch.unbind(emb, 1), dim=1)

# Method 3. Using view
emb.view(-1, 6)
```

Let's check the equivalence:

```python
emb.view(32, -1) == torch.cat(torch.unbind(emb, 1), dim=1)  
# tensor([[True, True, True, True, True, True], ...]) --> all True means equivalent
```

The `view` is the most concise and efficient. The feasibility of `view` lies in the Pytorch's internal storage mechanism for tensors. For details, see [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/).

## Build MLP

Now we have the input to the MLP, let's build it.

```python
# Build the first layer
n_hidden = 100
n_input = 6

W1 = torch.randn((n_input, n_hidden))
b1 = torch.randn((n_hidden,))
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

# Build the second layer
n_output = 27
W2 = torch.randn((n_hidden, n_output))
b2 = torch.randn((n_output,))
logits = h @ W2 + b2

# Generate the probabilities
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)

# Loss function
loss = - probs[torch.arange(X.shape[0]), Y].log().mean()
```

The whole process is the `forward pass` of the MLP.

Here introduce the `cross_entropy` function to generate the loss depending on probabilities.

```python
loss = F.cross_entropy(logits, Y)

# equivalent to
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
loss = - probs[torch.arange(X.shape[0]), Y].log().mean()
```

And according to `backpropagation`, we turn on the `requires_grad` flag for all the parameters. Now we rewrite the parameters initialization and forward pass.

```python
# for reproducibility
g = torch.Generator().manual_seed(2147483647)   # consistent with Andrej's settings 

# setting parameters
n_input = 6             # 3 characters * 2D embedding
n_hidden = 100
n_output = 27

C = torch.randn((27, 2), requires_grad=True, generator=g)
W1 = torch.randn((n_input, n_hidden), requires_grad=True, generator=g)
b1 = torch.randn((n_hidden,), requires_grad=True, generator=g)
W2 = torch.randn((n_hidden, n_output), requires_grad=True, generator=g)
b2 = torch.randn((n_output,), requires_grad=True, generator=g)

parameters = [C, W1, b1, W2, b2]        # collect all parameters

# embedding
emb = C[X]

# forward pass
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
```

## Training

Like before, we need `embedding` and `forward pass` to iterate the training process.

```python
# embedding
emb = C[X]

# forward pass
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)

# backward pass
for p in parameters:
    p.grad = None
loss.backward()

# update
for p in parameters:
    p.data -= 1.0 * p.grad
```

Let's train the model for 5000 iterations.

```python
for _ in range(5000):

    # embedding
    emb = C[X]

    # forward pass
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print(loss.item())  # trace the loss

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    for p in parameters:
        p.data -= 1.0 * p.grad
```

The loss is decreasing, means the model is learning. Yay~!

```text
19.505229949951172
17.08448028564453
15.776530265808105
14.833340644836426
14.002605438232422
13.253263473510742
12.57991886138916
11.983102798461914
11.470492362976074
11.05185604095459
10.709587097167969
10.407631874084473
10.127808570861816
9.864364624023438
9.614503860473633
9.376439094543457
9.148944854736328
8.931109428405762
8.722230911254883
8.521748542785645
8.329227447509766
8.144325256347656
7.966790676116943
7.796450614929199
7.633184909820557
...
2.436875343322754
2.4368624687194824
2.436849594116211
2.4368364810943604
2.436823844909668
```

Over 5000 iterations, the loss is decreasing from `19.505` to `2.311`. It's good, but we can do much better.

## Mini-Batch

The first optimization is to use `mini-batch` to speed up the training process.

```python
for _ in range(5000):

    # mini-batch construction
    ix = torch.randint(0, X.shape[0], (32,))    # 32 is the batch size

    # embedding
    emb = C[X[ix]]  # randomly select 32 samples

    # forward pass
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])   # use the related labels

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    for p in parameters:
        p.data -= 0.1 * p.grad
```

Training process becomes much much faster, but we need to evaluate the loss on the whole dataset, not just the mini-batch.

```python
# embedding
emb = C[X]

# forward pass
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
print(loss.item())
```

The loss is decreasing from `2.311` to `2.288`, and the training process is much faster. Amazing!

## Learning Rate

The second optimization is to use `learning rate` to control the update step.

In the previous training process, we choose learning rate `1.0` and `0.1` by experience or guess. This is not a good practice, in the other words, it's not a optimal (or ideal) learning rate.

How can we deal with it?

