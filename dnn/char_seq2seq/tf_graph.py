# The following code is extended from the original code developed by JayParks
# Source: https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder


class CharacterSeq2SeqModel(object):
    """ The following code is extended from the original code developed by JayParks
        Source: https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py

    I have made the following extensions: adding bidirectional RNNs, accuracy metrics,
    visualize different curves between training and validation set data,
    collecting runtime metadata (for profiling TF graphs),
    and other minor improvements.
    """

    def __init__(self, session, config, mode):
        self.session = session
        assert mode.lower() in ['train', 'decode']
        self.config = config
        self.mode = mode.lower()

        self.hidden_units = config['hidden_units']
        self.depth = config['depth']
        self.attention_type = config['attention_type']
        self.embedding_size = config['embedding_size']
        
        self.num_encoder_symbols = config['num_encoder_symbols']
        self.num_decoder_symbols = config['num_decoder_symbols']

        self.use_residual = config['use_residual']
        self.attn_input_feeding = config['attn_input_feeding']
        self.use_dropout = config['use_dropout']
        self.keep_prob = 1.0 - config['dropout_rate']

        self.padding_token_index = config['padding_token_index']
        self.dest_start_token_index = config['dest_start_token_index']
        self.dest_eos_token_index = config['dest_eos_token_index']
        
        self.max_dest_seq_length = config['max_dest_seq_length']
        self.max_src_seq_length = config['max_src_seq_length']
        
        self.optimizer = config['optimizer']
        self.learning_rate = config['learning_rate']
        self.max_gradient_norm = config['max_gradient_norm']
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.dtype = tf.float16 if config['use_fp16'] else tf.float32
        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')
        self.batch_size = config['batch_size']
        self.use_beamsearch_decode = config['use_beamsearch_decode']
        self.beam_width = config['beam_width']   
        
        if config['run_full_trace']:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        else:
            self.run_options = tf.RunOptions()
        self.run_metadata = tf.RunMetadata()
        
        if self.mode == 'decode':
            self.use_beamsearch_decode = True if self.beam_width > 1 else False
            self.max_decode_step = config['max_decode_step']

        if 'default_device' in config:
            self.default_device = config['default_device']
            print(" Setting default device to ", self.default_device)
            
            with tf.device(self.default_device):
                self.build_model()
        else:
            self.build_model()
    
    def merge_lstm_state_tuple(self,
                               first_lstm_tuple, 
                               second_lstm_state_tuple):
        """ Merge and reduce dimension back to the same"""
        projection_layer = Dense(units=self.hidden_units)
        h_new = tf.concat([first_lstm_tuple.h, second_lstm_state_tuple.h], axis=1)
        c_new = tf.concat([first_lstm_tuple.c, second_lstm_state_tuple.c], axis=1)
        return LSTMStateTuple(c=projection_layer(c_new),
                              h=projection_layer(h_new))

    def build_model(self):
        print("building model..")

        # Building encoder and decoder networks
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self):
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size, self.max_src_seq_length),
            name='encoder_inputs')

        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(self.batch_size,), name='encoder_inputs_length')

        if self.mode == 'train':
            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(
               dtype=tf.int32, 
               shape=(self.batch_size, self.max_dest_seq_length)
               , name='decoder_inputs')
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(self.batch_size,), name='decoder_inputs_length')

            self.decoder_inputs_train = self.decoder_inputs
            self.decoder_inputs_length_train = self.decoder_inputs_length
            self.decoder_inputs_length_masks = self.decoder_inputs_length
    
            # Maximum decoder time_steps in current batch
            self.max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)
            
            # Prepare decode target -> first padding extra
            padding_token = tf.ones(
                 shape=[self.batch_size, 1], dtype=tf.int32) * self.padding_token_index
            self.decoder_targets_train = tf.concat([self.decoder_inputs[:, 1:],
                                                    padding_token], axis=1)
            
            # Slice the target to match the max decoder length
            self.decoder_targets_train = self.decoder_targets_train[:,
                                                                    :self.max_decoder_length]

    def build_encoder(self):
        print("building encoder..")
        with tf.variable_scope('encoder'):
            # Building encoder_cell
            self.encoder_cell = self.build_encoder_cell()
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')
            
            # Initialize encoder_embeddings to have variance=1.
            initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3), dtype=self.dtype)
            self.encoder_embeddings = tf.get_variable(
                name='embedding', shape=[self.num_encoder_symbols, self.embedding_size],
                initializer=initializer, dtype=self.dtype)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=self.encoder_inputs)

            self.encoder_inputs_encoded = input_layer(self.encoder_inputs_embedded)
            (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.encoder_cell, cell_bw=self.encoder_cell, 
                inputs=self.encoder_inputs_encoded,
                sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                time_major=False)            
            self.encoder_outputs = Dense(self.hidden_units)(tf.concat([output_fw, output_bw], axis=1))
            self.encoder_last_state = [self.merge_lstm_state_tuple(
                                       output_state_fw[i], output_state_bw[i])
                                       for i in range(len(output_state_fw))]

    def build_decoder(self):
        print("building decoder and attention..")
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')
            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.num_decoder_symbols, name='output_projection')
            
            if self.mode == 'train':
                initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3), dtype=self.dtype)
                self.decoder_embeddings = tf.get_variable(
                        name='embedding', shape=[self.num_decoder_symbols, self.embedding_size], initializer=initializer, dtype=self.dtype)
                self.decoder_encoded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings, ids=self.decoder_inputs_train)
                self.decoder_inputs_encoded = input_layer(self.decoder_encoded)
                print(" Decoder input encoded is ", self.decoder_inputs_encoded.shape)
                
                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(
                    inputs=self.decoder_inputs_encoded,
                    sequence_length=self.decoder_inputs_length_train,
                    time_major=False, name='training_helper')

                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell, helper=training_helper,
                    initial_state=self.decoder_initial_state, output_layer=output_layer)

                (self.decoder_outputs_train, self.decoder_last_state_train,
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    swap_memory=True,
                    maximum_iterations=self.max_decoder_length))

                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output) 
                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                    name='decoder_pred_train')

                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_masks, 
                                         maxlen=self.max_decoder_length, dtype=self.dtype, name='masks')

                print("logits train shape is ", self.decoder_logits_train.shape)
                print("decoder_targets_train train shape is ", self.decoder_targets_train.shape)
                
                self.loss = tf.reduce_sum(seq2seq.sequence_loss(
                    logits=self.decoder_logits_train,
                    targets=self.decoder_targets_train,
                    weights=masks, average_across_timesteps=False,
                    average_across_batch=True,))

                # Compute predictions
                self.accuracy, self.accuracy_op = tf.metrics.accuracy(
                    labels=self.decoder_targets_train, 
                    predictions=self.decoder_pred_train,
                    name="accuracy")
                
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('teacher_forcing_accuracy', self.accuracy)
                
                # Contruct graphs for minimizing loss
                self.init_optimizer()

            elif self.mode == 'decode':
                self.decoder_embeddings = tf.get_variable(
                name='embedding', shape=[self.num_decoder_symbols, self.embedding_size], dtype=self.dtype)
        
                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.dest_start_token_index
                end_token = self.dest_eos_token_index

                def embed_and_input_proj(inputs):
                    encoded_input = tf.nn.embedding_lookup(self.decoder_embeddings, inputs)
                    return input_layer(encoded_input)

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=end_token,
                                                                    embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=self.decoder_initial_state,
                                                             output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately find the most likely translation
                    print("building beamsearch decoder..")
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(
                        cell=self.decoder_cell, embedding=embed_and_input_proj,
                        start_tokens=start_tokens, end_token=end_token,
                        initial_state=self.decoder_initial_state, beam_width=self.beam_width,
                        output_layer=output_layer,)

                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    swap_memory=True,
                    maximum_iterations=self.max_decode_step))

                if not self.use_beamsearch_decode:
                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

    def build_single_cell(self):
        cell_type = LSTMCell
        cell = cell_type(self.hidden_units)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder,)
        if self.use_residual:
            cell = ResidualWrapper(cell)
        return cell

    # Building encoder cell
    def build_encoder_cell(self):
        return MultiRNNCell([self.build_single_cell() for _ in range(self.depth)])

    # Building decoder cell and attention. Also returns decoder_initial_state
    def build_decoder_cell(self):
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length 
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:
            print("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        self.attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=self.hidden_units, memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length,) 
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=self.hidden_units, memory=encoder_outputs, 
                memory_sequence_length=encoder_inputs_length,)
 
        # Building decoder_cell
        self.decoder_cell_list = [
            self.build_single_cell() for _ in range(self.depth)]
        
        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(self.hidden_units, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state
        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode \
            else self.batch_size * self.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
          batch_size=batch_size, dtype=self.dtype)
        
        print(" Initial state for decoder cell batch size is ", batch_size)
               
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

    def init_optimizer(self):
        print("setting optimizer..")
        # Gradients and SGD update operation for training the model
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def save(self, sess, path, var_list=None, global_step=None):
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path=path)
        print('model saved at %s' % save_path)

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

    def train(self, sess, encoder_inputs, encoder_inputs_length, 
              decoder_inputs, decoder_inputs_length):
        """Run a train step of the model feeding the given inputs.
        Args:
          sess: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        output_feed = [self.updates,  # Update Op that does optimization
                       self.loss,  # Loss for current batch
                       self.accuracy,  # Accuracy for current batch
                       self.summary_op  # Training summary
                       ]
        
        # Computing accuracy in a separate session call
        # Why? http://ronny.rest/blog/post_2017_09_11_tf_metrics/
        sess.run(self.accuracy_op, input_feed) 
        
        outputs = sess.run(output_feed, input_feed, options=self.run_options, run_metadata=self.run_metadata)
                
        return outputs[1], outputs[2], outputs[3]  # loss, accuracy, summary

    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        """Run a evaluation step of the model feeding the given inputs.
        Args:
          sess: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        # Do not use Dropout during evaluation
        input_feed[self.keep_prob_placeholder.name] = 1.0

        # computing accuracy in a separate session call
        # Why? http://ronny.rest/blog/post_2017_09_11_tf_metrics/
        sess.run(self.accuracy_op, input_feed)
        
        output_feed = [self.loss,  # Loss for current batch
                       self.accuracy,  # Accuracy for current batch
                       self.summary_op  # Training summary
                       ]
        outputs = sess.run(output_feed, input_feed, options=self.run_options, run_metadata=self.run_metadata)
        return outputs[0], outputs[1], outputs[2]  # loss, accuracy, summary

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length, 
                                      decoder_inputs=None, decoder_inputs_length=None, 
                                      decode=True)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.decoder_pred_decode]
        outputs = sess.run(output_feed, input_feed)

        # GreedyDecoder: [batch_size, max_time_step]
        return outputs[0]  # BeamSearchDecoder: [batch_size, max_time_step, beam_width]

    def check_feeds(self, encoder_inputs, encoder_inputs_length, 
                    decoder_inputs, decoder_inputs_length, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                             "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                                 "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                                 "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))
        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed 
