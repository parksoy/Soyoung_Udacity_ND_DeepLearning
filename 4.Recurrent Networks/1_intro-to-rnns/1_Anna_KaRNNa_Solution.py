#####################################
# # Anna KaRNNa
#####################################
# In this notebook, I'll build a character-wise RNN trained on Anna Karenina,
# one of my all-time favorite books.
# It'll be able to generate new text based on the text from the book.
# This network is based off of Andrej Karpathy's [post on RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
#  and [implementation in Torch](https://github.com/karpathy/char-rnn). Also, some
# information [here at r2rt](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html)
# and from [Sherjil Ozair](https://github.com/sherjilozair/char-rnn-tensorflow) on GitHub. Below is the general architecture of the character-wise RNN.
#
# <img src="assets/charseq.jpeg" width="500">

import time
from collections import namedtuple
import numpy as np
import tensorflow as tf

# First we'll load the text file and convert it into integers for our network to use. Here I'm
# creating a couple dictionaries to convert the characters to and from integers. Encoding
# the characters as integers makes it easier to use as input in the network.

with open('anna.txt', 'r') as f:
    text=f.read()

vocab = sorted(set(text))    # ['\n', ' ', '!', '"', '$', '%',
vocab_to_int = {c: i for i, c in enumerate(vocab)} #{'w': 79, '1': 16, '7': 22, 't': 76, 'O'
int_to_vocab = dict(enumerate(vocab)) #{0: '\n', 1: ' ', 2: '!', 3: '"',

print("text=n", text[:100])
print("vocab=\n", vocab)
print("vocab_to_int=\n", vocab_to_int)
print("int_to_vocab=\n", int_to_vocab)

encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32) #array([31, 64, 57, 72,

# Let's check out the first 100 characters, make sure everything is peachy. According to the
# [American Book Review](http://americanbookreview.org/100bestlines.asp), this is the 6th best first line of a book ever.

text[:100]

# And we can see the characters encoded as integers.

encoded[:100]

# Since the network is working with individual characters, it's similar to a classification problem in which we
#  are trying to predict the next character from the previous text.  Here's how many 'classes' our network has to pick from.

len(vocab)
#####################################
# ## Making training mini-batches
#####################################
# Here is where we'll make our mini-batches for training. Remember that we want our batches to be multiple
# sequences of some desired number of sequence steps. Considering a simple example, our batches would look like this:
# <img src="assets/sequence_batching@1x.png" width=500px>
# <br>
#
# We start with our text encoded as integers in one long array in `encoded`. Let's create a function
# that will give us an iterator for our batches. I like using
# [generator functions](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/)
# to do this. Then we can pass `encoded` into this function and get our batch generator.
# The first thing we need to do is discard some of the text so we only have completely full batches.
# Each batch contains $N \times M$ characters, where $N$ is the batch size (the number of sequences)
# and $M$ is the number of steps. Then, to get the total number of batches, $K$, we can make from the
# array `arr`, you divide the length of `arr` by the number of characters per batch. Once you know the
# number of batches, you can get the total number of characters to keep from `arr`, $N * M * K$.
# After that, we need to split `arr` into $N$ sequences. You can do this using `arr.reshape(size)`
# where `size` is a tuple containing the dimensions sizes of the reshaped array. We know we want $N$
# sequences (`batch_size` below), let's make that the size of the first dimension. For the second dimension,
# you can use `-1` as a placeholder in the size, it'll fill up the array with the appropriate data for you.
# After this, you should have an array that is $N \times (M * K)$.
#
# Now that we have this array, we can iterate through it to get our batches. The idea is each batch is a $N \times
# M$ window on the $N \times (M * K)$ array. For each subsequent batch, the window moves over by `n_steps`.
# We also want to create both the input and target arrays. Remember that the targets are the inputs shifted over one character.
#
# The way I like to do this window is use `range` to take steps of size `n_steps` from $0$ to `arr.shape[1]`,
# the total number of steps in each sequence. That way, the integers you get from `range` always point to the
# start of a batch, and each window is `n_steps` wide.

def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr)//chars_per_batch

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * chars_per_batch]

    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n+n_steps]
        # The targets, shifted by one
        y_temp = arr[:, n+1:n+n_steps+1]

        # For the very last batch, y will be one character short at the end of
        # the sequences which breaks things. To get around this, I'll make an
        # array of the appropriate size first, of all zeros, then add the targets.
        # This will introduce a small artifact in the last batch, but it won't matter.
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:,:y_temp.shape[1]] = y_temp

        yield x, y


# Now I'll make my data sets and we can check out what's going on here. Here I'm going to use a
# batch size of 10 and 50 sequence steps.

batches = get_batches(encoded, 10, 50)
x, y = next(batches)

print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])


# If you implemented `get_batches` correctly, the above output should look something like
# ```
# x
#  [[55 63 69 22  6 76 45  5 16 35]
#  [ 5 69  1  5 12 52  6  5 56 52]
#  [48 29 12 61 35 35  8 64 76 78]
#  [12  5 24 39 45 29 12 56  5 63]
#  [ 5 29  6  5 29 78 28  5 78 29]
#  [ 5 13  6  5 36 69 78 35 52 12]
#  [63 76 12  5 18 52  1 76  5 58]
#  [34  5 73 39  6  5 12 52 36  5]
#  [ 6  5 29 78 12 79  6 61  5 59]
#  [ 5 78 69 29 24  5  6 52  5 63]]
#
# y
#  [[63 69 22  6 76 45  5 16 35 35]
#  [69  1  5 12 52  6  5 56 52 29]
#  [29 12 61 35 35  8 64 76 78 28]
#  [ 5 24 39 45 29 12 56  5 63 29]
#  [29  6  5 29 78 28  5 78 29 45]
#  [13  6  5 36 69 78 35 52 12 43]
#  [76 12  5 18 52  1 76  5 58 52]
#  [ 5 73 39  6  5 12 52 36  5 78]
#  [ 5 29 78 12 79  6 61  5 59 63]
#  [78 69 29 24  5  6 52  5 63 76]]
#  ```
#  although the exact numbers will be different. Check to make sure the data is shifted
# over one step for `y`.

#####################################
# ## Building the model
#####################################
# Below is where you'll build the network. We'll break it up into parts so it's easier to
# reason about each bit. Then we can connect them up into the whole network.
#
# <img src="assets/charRNN.png" width=500px>
#####################################
# ### Inputs
#####################################
# First off we'll create our input placeholders. As usual we need placeholders for the training
# data and the targets. We'll also create a placeholder for dropout layers called `keep_prob`.

def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout

        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch

    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob

#####################################
# ### LSTM Cell
#####################################
# Here we will create the LSTM cell we'll use in the hidden layer. We'll use this cell as a
# building block for the RNN. So we aren't actually defining the RNN here, just the type of
# cell we'll use in the hidden layer.
#
# We first create a basic LSTM cell with
#
# ```python
# lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
# ```
#
# where `num_units` is the number of units in the hidden layers in the cell. Then we can add
#  dropout by wrapping it with
#
# ```python
# tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
# ```
# You pass in a cell and it will automatically add dropout to the inputs or outputs. Finally,
# we can stack up the LSTM cells into layers with
# [`tf.contrib.rnn.MultiRNNCell`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/rnn/MultiRNNCell).
# With this, you pass in a list of cells and it will send the output of one cell into the next cell. Previously with TensorFlow 1.0, you could do this
#
# ```python
# tf.contrib.rnn.MultiRNNCell([cell]*num_layers)
# ```
#
# This might look a little weird if you know Python well because this will create a list of the same `cell` object.
# However, TensorFlow 1.0 will create different weight matrices for all `cell` objects.
#  But, starting with
#  TensorFlow 1.1 you actually need to create new cell objects in the list. To get it to work in TensorFlow 1.1, it should look like
#
# ```python
# def build_cell(num_units, keep_prob):
#     lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
#     drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
#
#     return drop
#
# tf.contrib.rnn.MultiRNNCell([build_cell(num_units, keep_prob) for _ in range(num_layers)])
# ```
#
# Even though this is actually multiple LSTM cells stacked on each other, you can treat the multiple
# layers as one cell.
#
# We also need to create an initial cell state of all zeros. This can be done like so
#
# ```python
# initial_state = cell.zero_state(batch_size, tf.float32)
# ```
#
# Below, we implement the `build_lstm` function to create these LSTM cells and the initial state.

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.

        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size

    '''
    ### Build the LSTM Cell

    def build_cell(lstm_size, keep_prob):
        # Use a basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop


    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state

#####################################
# ### RNN Output
#####################################
# Here we'll create the output layer. We need to connect the output of the RNN cells to
# a full connected layer with a softmax output. The softmax output gives us a probability
# distribution we can use to predict the next character.
# If our input has batch size $N$, number of steps $M$, and the hidden layer has $L$ hidden units,
# then the output is a 3D tensor with size $N \times M \times L$. The output of each LSTM cell
# has size $L$, we have $M$ of them, one for each sequence step, and we have $N$ sequences.
# So the total size is $N \times M \times L$.
#
# We are using the same fully connected layer, the same weights, for each of the outputs. Then,
# to make things easier, we should reshape the outputs into a 2D tensor with shape $(M * N) \times L$.
# That is, one row for each sequence and step, where the values of each row are the output from the LSTM cells.
#
# One we have the outputs reshaped, we can do the matrix multiplication with the weights.
# We need to wrap the weight and bias variables in a variable scope with `tf.variable_scope(scope_name)`
# because there are weights being created in the LSTM cells. TensorFlow will throw an error if the
# weights created here have the same names as the weights created in the LSTM cells, which they will
#  be default. To avoid this, we wrap the variables in a variable scope so we can give them unique names.


def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.

        Arguments
        ---------

        x: Input tensor
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer

    '''

    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # That is, the shape should be batch_size*num_steps rows by lstm_size columns
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])

    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(x, softmax_w) + softmax_b

    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits

#####################################
# ### Training loss
#####################################
# Next up is the training loss. We get the logits and targets and calculate the softmax
# cross-entropy loss. First we need to one-hot encode the targets, we're getting them as
# encoded characters. Then, reshape the one-hot targets so it's a 2D tensor with size $(M*N) \times
#  C$ where $C$ is the number of classes/characters we have. Remember that we reshaped
# the LSTM outputs and ran them through a fully connected layer with $C$ units. So our
#  logits will also have size $(M*N) \times C$.
# Then we run the logits and targets through `tf.nn.softmax_cross_entropy_with_logits` and
# find the mean to get the loss.

def build_loss(logits, targets, lstm_size, num_classes):
    ''' Calculate the loss from the logits and the targets.

        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        lstm_size: Number of LSTM hidden units
        num_classes: Number of classes in targets

    '''

    # One-hot encode targets and reshape to match logits, one row per batch_size per step
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

#####################################
# ### Optimizer
#####################################
# Here we build the optimizer. Normal RNNs have have issues gradients exploding and
# disappearing. LSTMs fix the disappearance problem, but the gradients can still
# grow without bound. To fix this, we can clip the gradients above some threshold.
# That is, if a gradient is larger than that threshold, we set it to the threshold.
# This will ensure the gradients never grow overly large. Then we use an AdamOptimizer
# for the learning step.

def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.

        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer

    '''

    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

#####################################
# ### Build the network
#####################################
# Now we can put all the pieces together and build a class for the network. To actually
# run data through the LSTM cells, we will use
# [`tf.nn.dynamic_rnn`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/nn/dynamic_rnn).
# This function will pass the hidden and cell states across LSTM cells appropriately for us.
# It returns the outputs for each LSTM cell at each step for each sequence in the mini-batch.
# It also gives us the final LSTM state. We want to save this state as `final_state` so
# we can pass it to the first LSTM cell in the the next mini-batch run. For `tf.nn.dynamic_rnn`,
# we pass in the cell and initial state we get from `build_lstm`, as well as our input sequences.
#Also, we need to one-hot encode the inputs before going into the RNN.

class CharRNN:

    def __init__(self, num_classes, batch_size=64, num_steps=50,
                       lstm_size=128, num_layers=2, learning_rate=0.001,
                       grad_clip=5, sampling=False):

        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()

        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        # Build the LSTM cell
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        ### Run the data through the RNN layers
        # First, one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)

        # Run each sequence step through the RNN and collect the outputs
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

#####################################
# ## Hyperparameters
#####################################
# Here I'm defining the hyperparameters for the network.
# * `batch_size` - Number of sequences running through the network in one pass.
# * `num_steps` - Number of characters in the sequence the network is trained on.
# Larger is better typically, the network will learn more long range dependencies.
# But it takes longer to train. 100 is typically a good number here.
# * `lstm_size` - The number of units in the hidden layers.
# * `num_layers` - Number of hidden LSTM layers to use
# * `learning_rate` - Learning rate for training
# * `keep_prob` - The dropout keep probability when training. If you're network is
# overfitting, try decreasing this.
# Here's some good advice from Andrej Karpathy on training the network. I'm going to
# copy it in here for your benefit, but also link to
# [where it originally came from](https://github.com/karpathy/char-rnn#tips-and-tricks).

######################################
# > ## Tips and Tricks
#####################################
# >### Monitoring Validation Loss vs. Training Loss
# >If you're somewhat new to Machine Learning or Neural Networks it can take a bit of
# expertise to get good models. The most important quantity to keep track of is the
# difference between your training loss (printed during training) and the validation
# loss (printed once in a while when the RNN is run on the validation data
# (by default every 1000 iterations)). In particular:
#
# > - If your training loss is much lower than validation loss then this means the network
# might be **overfitting**. Solutions to this are to decrease your network size, or to
# increase dropout. For example you could try dropout of 0.5 and so on.
# > - If your training/validation loss are about equal then your model is **underfitting**.
# Increase the size of your model (either number of layers or the raw number of neurons per layer)

#####################################
# > ### Approximate number of parameters
#####################################
# > The two most important parameters that control the model are `lstm_size` and `num_layers`.
# I would advise that you always use `num_layers` of either 2/3. The `lstm_size` can be
# adjusted based on how much data you have. The two important quantities to keep track of here are:
#
# > - The number of parameters in your model. This is printed when you start training.
# > - The size of your dataset. 1MB file is approximately 1 million characters.
#
# >These two should be about the same order of magnitude. It's a little tricky to tell.
# Here are some examples:
#
# > - I have a 100MB dataset and I'm using the default parameter settings (which currently print
# 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to
# heavily underfit. I am thinking I can comfortably afford to make `lstm_size` larger.
# > - I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm
# carefully monitoring my validation loss. If it's larger than my training loss then I may want to
# try to increase dropout a bit and see if that helps the validation loss.

#####################################
# > ### Best models strategy
#####################################
# >The winning strategy to obtaining very good models (if you have the compute time) is
# to always err on making the network larger (as large as you're willing to wait for it to
# compute) and then try different dropout values (between 0,1). Whatever model has the best
#  validation performance (the loss, written in the checkpoint filename, low is good) is the
# one you should use in the end.
#
# >It is very common in deep learning to run many different models with many different
# hyperparameter settings,
# and in the end take whatever checkpoint gave the best validation performance.
# >By the way, the size of your training and validation splits are also parameters. Make
# sure you have a decent amount of data in your validation set or otherwise the validation
# performance will be noisy and not very informative.

batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability

#####################################
# ## Time for training
#####################################
# This is typical training code, passing inputs and targets into the network, then running the
# optimizer. Here we also get back the final LSTM state for the mini-batch. Then, we pass that
# state back into the network so the next batch can continue the state from the previous batch.
# And every so often (set by `save_every_n`) I save a checkpoint.
# Here I'm saving checkpoints with the format
# `i{iteration number}_l{# hidden layer units}.ckpt`

epochs = 20
# Print losses every N interations
print_every_n = 50

# Save every N iterations
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Use the line below to load a checkpoint and resume training
    #saver.restore(sess, 'checkpoints/______.ckpt')
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                 feed_dict=feed)
            if (counter % print_every_n == 0):
                end = time.time()
                print('Epoch: {}/{}... '.format(e+1, epochs),
                      'Training Step: {}... '.format(counter),
                      'Training loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

#####################################
# #### Saved checkpoints
#####################################
# Read up on saving and loading checkpoints here: https://www.tensorflow.org/programmers_guide/variables

tf.train.get_checkpoint_state('checkpoints')

#####################################
# ## Sampling
#####################################
# Now that the network is trained, we'll can use it to generate new text. The idea is
# that we pass in a
# character, then the network will predict the next character. We can use the new one,
# to predict the next one. And we keep doing this to generate all new text. I also included
# some functionality to prime the network with some text by passing in a string and
#  building up a state from that.
# The network gives us predictions for each character. To reduce noise and make things a
# little less random, I'm going to only choose a new character from the top N most likely characters.
#
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return ''.join(samples)

# Here, pass in the path to a checkpoint and sample from the network.
tf.train.latest_checkpoint('checkpoints')

checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="Far")
print(samp)

checkpoint = 'checkpoints/i200_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)

checkpoint = 'checkpoints/i600_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)

checkpoint = 'checkpoints/i1200_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)
