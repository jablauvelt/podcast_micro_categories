import json, os, re, shutil, sys, time
import a3utils

TF_SAVEDIR = "tf_saved"
checkpoint_filename = os.path.join(TF_SAVEDIR, "rnnlm")
trained_filename = os.path.join(TF_SAVEDIR, "rnnlm_trained")

def run_epoch(lm, session, batch_iterator,
              train=False, verbose=False,
              tick_s=10, learning_rate=0.1):
    start_time = time.time()
    tick_time = start_time  # for showing status
    total_cost = 0.0  # total cost, summed over all words
    total_batches = 0
    total_words = 0

    verbose=True
    
    if train:
        train_op = lm.train_step_
        use_dropout = True
        loss = lm.train_loss_
    else:
        train_op = tf.no_op()
        use_dropout = False  # no dropout at test time
        loss = lm.loss_  # true loss, if train_loss is an approximation

    for i, (w, y) in enumerate(batch_iterator):
        cost = 0.0
        # At first batch in epoch, get a clean intitial state.
        if i == 0:
            h = session.run(lm.initial_h_, {lm.input_w_: w})

        #### YOUR CODE HERE ####
        feed_dict = { lm.input_w_: w, 
                      lm.target_y_: y, 
                      lm.initial_h_ : h, 
                      lm.learning_rate_ : learning_rate, 
                      lm.use_dropout_ : use_dropout 
                    }
        
        h, cost, _ = session.run( [lm.final_h_, loss, train_op], feed_dict=feed_dict )
        
        #### END(YOUR CODE) ####
        total_cost += cost
        total_batches = i + 1
        total_words += w.size  # w.size = batch_size * max_time

        ##
        # Print average loss-so-far for epoch
        # If using train_loss_, this may be an underestimate.
        if verbose and (time.time() - tick_time >= tick_s):
            avg_cost = total_cost / total_batches
            avg_wps = total_words / (time.time() - start_time)
            print "[batch %d]: seen %d words at %d wps, loss = %.3f" % (
                i, total_words, avg_wps, avg_cost)
            tick_time = time.time()  # reset time ticker

    return total_cost / total_batches, h

def score_dataset(lm, session, ids, name="Data", is_final=False):
    # For scoring, we can use larger batches to speed things up.
    final_h = None
    bi = a3utils.batch_generator(ids, batch_size=100, max_time=100)
    cost, h = run_epoch(lm, session, bi, 
                     learning_rate=1.0, train=False, 
                     verbose=True, tick_s=3600)
    print "%s: avg. loss: %.03f  (perplexity: %.02f)" % (name, cost, np.exp(cost))
    if is_final:
        final_h = h[0][1]
    return final_h
    
    
def execute_rnnlm(model_params, training_params, vocab, train_ids, test_ids):
    # Will print status every this many seconds
    print_interval = 15

    # Training parameters
    max_time = training_params['max_time']
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate']
    num_epochs = training_params['num_epochs']

    # Clear old log directory
    shutil.rmtree("tf_summaries", ignore_errors=True)

    lm = RNNLM(**model_params)
    lm.BuildCoreGraph()
    lm.BuildTrainGraph()

    # Explicitly add global initializer and variable saver to LM graph
    with lm.graph.as_default():
        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver()

    # Clear old log directory
    shutil.rmtree(TF_SAVEDIR, ignore_errors=True)
    if not os.path.isdir(TF_SAVEDIR):
        os.makedirs(TF_SAVEDIR)

    with tf.Session(graph=lm.graph) as session:
        # Seed RNG for repeatability
        tf.set_random_seed(42)

        session.run(initializer)
        
        final_h = None
        
        for epoch in xrange(1,num_epochs+1):
            t0_epoch = time.time()
            bi = a3utils.batch_generator(train_ids, batch_size, max_time)
            print "[epoch %d] Starting epoch %d" % (epoch, epoch)
            #### YOUR CODE HERE ####
            # Run a training epoch.

            v = run_epoch(lm, session, bi,
                  train=True, verbose=True,
                  tick_s=print_interval, learning_rate=learning_rate)

            #### END(YOUR CODE) ####
            print "[epoch %d] Completed in %s" % (epoch, a3utils.pretty_timedelta(since=t0_epoch))

            # Save a checkpoint
            saver.save(session, checkpoint_filename, global_step=epoch)

            ##
            # score_dataset will run a forward pass over the entire dataset
            # and report perplexity scores. This can be slow (around 1/2 to 
            # 1/4 as long as a full epoch), so you may want to comment it out
            # to speed up training on a slow machine. Be sure to run it at the 
            # end to evaluate your score.
            print ("[epoch %d]" % epoch),
            final_h = score_dataset(lm, session, train_ids, name="Train set")
            print ("[epoch %d]" % epoch),
            final_h = score_dataset(lm, session, test_ids, name="Test set", is_final=(epoch==num_epochs))
            print ""

        return final_h
        # Save final model
        saver.save(session, trained_filename)   
        
# ----------------

import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    return cell


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper

class RNNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder_with_default(
                0.1, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 5.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core RNNLM graph, needed for any use of the model.

        This should include:
        - Placeholders for input tensors (input_w_, initial_h_, target_y_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).

        You shouldn't include training or sampling functions here; you'll do
        this in BuildTrainGraph and BuildSampleGraph below.

        We give you some starter definitions for input_w_ and target_y_, as
        well as a few other tensors that might help. We've also added dummy
        values for initial_h_, logits_, and loss_ - you should re-define these
        in your code as the appropriate tensors.

        See the in-line comments for more detail.
        """
        # Input ids, with dynamic shape depending on input.
        # Should be shape [batch_size, max_time] and contain integer word indices.
        self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")

        # Initial hidden state. You'll need to overwrite this with cell.zero_state
        # once you construct your RNN cell.
        self.initial_h_ = None

        # Final hidden state. You'll need to overwrite this with the output from
        # tf.nn.dynamic_rnn so that you can pass it in to the next batch (if
        # applicable).
        self.final_h_ = None

        # Output logits, which can be used by loss functions or for prediction.
        # Overwrite this with an actual Tensor of shape [batch_size, max_time]
        self.logits_ = None

        # Should be the same shape as inputs_w_
        self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

        # Get dynamic shape info from inputs
        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_w_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_w_)[1]

        # Get sequence length from input_w_.
        # TL;DR: pass this to dynamic_rnn.
        # This will be a vector with elements ns[i] = len(input_w_[i])
        # You can override this in feed_dict if you want to have different-length
        # sequences in the same batch, although you shouldn't need to for this
        # assignment.
        self.ns_ = tf.tile([self.max_time_], [self.batch_size_, ], name="ns")

        #### YOUR CODE HERE ####

        # Construct embedding layer
        with tf.name_scope("embedding_layer"):
            self.W_in_ = tf.Variable( tf.random_uniform( [self.V, self.H], 0.0, 1.0), name="W_in")
            x_ = tf.nn.embedding_lookup(self.W_in_, self.input_w_ )
        
        # Construct RNN/LSTM cell and recurrent layer (hint: use tf.nn.dynamic_rnn)
        with tf.name_scope("recurrent_layer"):
            self.cell_ = MakeFancyRNNCell( self.H, self.dropout_keep_prob_, self.num_layers)
            self.initial_h_ = self.cell_.zero_state( self.batch_size_, tf.float32 )
            self.out_, self.final_h_ = tf.nn.dynamic_rnn( self.cell_, x_, self.ns_, self.initial_h_)
            
            self.W_out_ = tf.Variable( tf.random_uniform( [self.H, self.V], 0.0, 1.0), name="W_out")
            self.b_out_ = tf.Variable( tf.zeros([self.V,], dtype=tf.float32, name="b_out"))
            
        # Softmax output layer, over vocabulary. Just compute logits_ here.
        # Hint: the matmul3d function will be useful here; it's a drop-in
        # replacement for tf.matmul that will handle the "time" dimension
        # properly.
        
        with tf.name_scope("output_layer"):
            self.logits_ = matmul3d( self.out_, self.W_out_) + self.b_out_
        
        with tf.name_scope("cost_function"):
            per_example_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_, self.target_y_, name="per_example_loss")
            self.loss_ = tf.reduce_mean(per_example_loss_, name="loss")

        # Loss computation (true loss, for prediction)
        #### END(YOUR CODE) ####

    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.

        You should define:
        - train_loss_ : sampled softmax loss, for training
        - train_step_ : a training op that can be called once per batch

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).
        """
        # Initializer step
        init_ = tf.global_variables_initializer()
                
        #### YOUR CODE HERE ####
        # Define approximate loss function.        
        # Note: self.softmax_ns (i.e. k=200) is already defined; use that as the        
        # number of samples.            
        # Loss computation (sampled, for training)
        per_example_train_loss_ = tf.nn.sampled_softmax_loss( 
                                                weights=tf.transpose(self.W_out_), 
                                                biases=self.b_out_, 
                                                labels=tf.reshape(self.target_y_, [-1,1] ),
                                                inputs=tf.reshape(self.out_, [-1, self.H] ), 
                                                num_sampled=self.softmax_ns, 
                                                num_classes=self.V,
                                                name="per_example_sampled_softmax_loss")
        
        self.train_loss_ = tf.reduce_mean(per_example_train_loss_, name="sampled_softmax_loss")

        # Define optimizer and training op
        optimizer_ = tf.train.AdagradOptimizer(self.learning_rate_)
        self.train_step_  = optimizer_.minimize(self.train_loss_)


        #### END(YOUR CODE) ####

    @with_self_graph
    def BuildSamplerGraph(self):
        """Construct the sampling ops.

        You should define pred_samples_ to be a Tensor of integer indices for
        sampled predictions for each batch element, at each timestep.

        Hint: use tf.multinomial, along with a couple of calls to tf.reshape
        """
        
        #### YOUR CODE HERE ####
     

        logits_2d = tf.reshape(self.logits_, [-1,self.V])
        self.pred_samples_ = tf.reshape( tf.multinomial( logits_2d, self.softmax_ns), [self.batch_size_, -1, 1] )

        #### END(YOUR CODE) ####


