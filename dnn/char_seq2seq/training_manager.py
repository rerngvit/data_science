"""Overall controlling code for the training process."""

import math
import tensorflow as tf
import time
from data_iterator import DataFeedIterator
from tf_graph import CharacterSeq2SeqModel
import numpy as np
import datetime


class TrainingManager(object):
    """ This class controls the training process by owning the model internally.
    """

    # Frequency of running the validation epoch (number of training epochs)
    VALIDATE_FREQ = 1     
    # Number of steps to display results
    DISPLAY_FREQ = 10
    # Number of epochs to tolerate score not improving
    EARLY_STOPPING_TOLERANCE_COUNT = 2
    
    def __init__(self, lang_data_preprocessor, src_data_path, dest_data_path, batch_size):
        """

        Arguments:
            lang_data_preprocessor: an object of language preprocessors to be used by this manager
            src_data_path: a file path for a CSV file for the source data
            dest_data_path: a file path for a CSV file for the destination data
            batch_size: the batch size to be used during the training
        """

        self.lang_data_preprocessor = lang_data_preprocessor
        self.hist_score = []
        self.patience_cnt = 0
        self.src_data_path = src_data_path
        self.dest_data_path = dest_data_path 
        self.batch_size = batch_size

        # Creating src_texts, dest_texts
        self.vectorize_data()
        
        # Preparing train / test / validation
        self.prepare_data()

    def vectorize_data(self):
        """ Vectorize the input data (from raw texts into dataframes) """

        self.src_texts = []
        self.dest_texts = []

        src_lines = open(self.src_data_path, encoding='utf-8')
        dest_lines = open(self.dest_data_path, encoding='utf-8')

        for src, dest in zip(src_lines, dest_lines):
            if len(src) <= self.lang_data_preprocessor.MAX_CHARACTER_LENGTH and\
                            len(dest) <= self.lang_data_preprocessor.MAX_CHARACTER_LENGTH:
                src = self.lang_data_preprocessor.preprocess(src) +\
                       self.lang_data_preprocessor.EOS_character
                dest = self.lang_data_preprocessor.SEQ_START_CHARACTER + \
                    self.lang_data_preprocessor.preprocess(dest) +\
                    self.lang_data_preprocessor.EOS_character
                self.src_texts.append(src)
                self.dest_texts.append(dest)

        self.src_texts = self.src_texts[:self.lang_data_preprocessor.MAX_NUMBER_OF_SAMPLES]
        self.dest_texts = self.dest_texts[:self.lang_data_preprocessor.MAX_NUMBER_OF_SAMPLES]

    def prepare_data(self):
        # Splitting train /test/ validation set
        from sklearn.model_selection import train_test_split
        self.src_texts_train, self.src_texts_test, \
            self.dest_texts_train, self.dest_texts_test = train_test_split(
                self.src_texts, self.dest_texts, test_size=0.2)
        self.src_texts_train, self.src_texts_valid, \
            self.dest_texts_train, self.dest_texts_valid = train_test_split(
                self.src_texts_train, self.dest_texts_train, test_size=0.3)

        # Construct Data feed iterator
        self.train_set = DataFeedIterator(
            lang_preprocessor=self.lang_data_preprocessor,
            src_texts=self.src_texts_train, dest_texts=self.dest_texts_train,
            batch_size=self.batch_size)
        self.valid_set = DataFeedIterator(lang_preprocessor=self.lang_data_preprocessor,
                                          src_texts=self.src_texts_valid, dest_texts=self.dest_texts_valid,
                                          batch_size=self.batch_size)
        self.test_set = DataFeedIterator(
            lang_preprocessor=self.lang_data_preprocessor,
            src_texts=self.src_texts_test, dest_texts=self.dest_texts_test,
            batch_size=self.batch_size)

    def reset_hist_score(self):
        """ Resetting the historical score (for early stopping implementation).
        This reset is needed for a new configuration to evaluate.
        """

        self.hist_score = []
        self.patience_cnt = 0

    def should_early_stop(self,
                          cost_score,
                          min_delta=0.01):
        """Decision function whether to execute early stopping or not

        Arguments:
            cost_score: the new cost score receive from the training function
            min_delta: the minimum delta to consider a valid reduction

        Returns:
            A boolean whether early stopping should be executed.
        """

        if math.isnan(cost_score):
            print("Cost score is NaN value => stopping now")
            return True
        else:
            print("Cost score is not Nan")
        print("Hist cost score is %s" % self.hist_score)
        if len(self.hist_score) >= 1:
            # Checking whether historical score
            if min(self.hist_score) - cost_score >= min_delta:
                self.patience_cnt = 0
                print(" Resetting patient count")
            else:
                self.patience_cnt += 1 
                print(" Increasing patient count")
        else:
            print(" no history yet -> not increasing score ")
            
        # Append last score
        self.hist_score.append(cost_score)

        return self.patience_cnt > TrainingManager.EARLY_STOPPING_TOLERANCE_COUNT
    
    def single_epoch_execution(self, model, session,
                               dataset, epoch_type,
                               log_writer, running_vars_initializer,
                               display_freq):
        """Execute the single epoch (full dataset walk through)

        Arguments:
            model: the CharacterSeq2Seq model object
            session: the Tensorflow session object to be used
            dataset: the dataset to operate on {"train", "validation", "test"}
            epoch_type: one of {"Train", "Validation"}
            log_writer: the tensorflow log_writer object to write for later on visualize on
                Tensorboard
            running_vars_initializer: initializer for local variables. In this case, it is
                needed to maintain the accuracy variable
            display_freq: the frequency of displaying output (in terms of number of epochs)
        """

        # epoch_type is either Train or Validation only
        assert(epoch_type == "Train" or epoch_type == "Validation")
        step_time, loss = 0.0, 0.0
        agg_loss, agg_accuracy = [], []
        samples_seen = 0
        start_time = time.time()
        
        # Reset running variables before each epoch
        session.run(running_vars_initializer)
        
        # Loop over samples from the dataset
        print('Execution an epoch: type = ', epoch_type)
        for step, (source, source_len, dest, dest_len) in enumerate(dataset): 
            if epoch_type == "Train":
                step_loss, teacher_forcing_accuracy, summary = model.train(
                    session, encoder_inputs=source, encoder_inputs_length=source_len,
                    decoder_inputs=dest, decoder_inputs_length=dest_len)
            else:
                step_loss, teacher_forcing_accuracy, summary = model.eval(
                    session, encoder_inputs=source, encoder_inputs_length=source_len,
                    decoder_inputs=dest, decoder_inputs_length=dest_len)
            
            if math.isnan(step_loss) or math.isnan(teacher_forcing_accuracy):
                print(" For epoch type", epoch_type)
                print(" having step_loss = ", step_loss, "; teacher-forcing accuracy", teacher_forcing_accuracy)
                # If we encounter NaN, stop after a single step (instead of wait until the end)
                return float('nan'), float('nan')
            
            loss += float(step_loss) / display_freq
            samples_seen += float(source.shape[0])  # batch_size of samples

            if step % display_freq == 0:
                now = datetime.datetime.now()
                epoch = model.global_epoch_step.eval(session=session)
                avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                time_elapsed = time.time() - start_time
                step_time = time_elapsed / display_freq  # average time spent/step
                samples_per_sec = samples_seen / time_elapsed        
                print('[',  epoch_type, ']',
                      'Epoch', epoch, 
                      'Step',  step,
                      'Teacher-Force Acc {0:.2f}'.format(teacher_forcing_accuracy),
                      'Loss {0:.2f}'.format(loss),                  
                      'Perplex {0:.2f}'.format(avg_perplexity), 
                      'Elapsed Time {0:.2f}'.format(step_time),
                      '{0:.2f} samples/s'.format(samples_per_sec), 
                      'Cur time: ', now.strftime("%Y-%m-%d %H:%M"),
                      '\n')

                agg_loss.append(loss)
                agg_accuracy.append(teacher_forcing_accuracy)

                loss = 0
                samples_seen = 0
                start_time = time.time()  # reset the clock for the next measurement

                # Record training summary for the current batch
                
                log_writer.add_run_metadata(
                    model.run_metadata, '%s-epoch-%d-step-%d' % (epoch_type, epoch, step))
                log_writer.add_summary(summary, step)

        print(" Finishing an epoch type ", epoch_type, " with agg_loss = ", loss)

        return np.array(agg_loss).mean(), np.array(agg_accuracy).mean()

    def fit_eval_dnn(self, config):
        """Fitting and evaluation function according to the provided configuration

        Arguments:
            config: a dictionary containing configuration parameter

        Returns:
            the best evaluation score according to the configuration
        """

        # Initialize the best metric to infinity
        best_valid_metric = float('inf')
        tf.reset_default_graph()  # to clean out all the variables to allow for rerunning the model
        self.reset_hist_score()
        
        graph = tf.Graph()
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(graph=graph, config=tf_config)

        with graph.as_default(), sess.as_default():
            model = CharacterSeq2SeqModel(sess, config, 'train')
            log_writer_train = tf.summary.FileWriter(config['model_base_dir'] +
                                                     "/tf_boards/plot_train", graph=sess.graph)
            log_writer_val = tf.summary.FileWriter(config['model_base_dir'] +
                                                   "/tf_boards/plot_val", graph=sess.graph)

            # Initializing global variables
            sess.run(tf.global_variables_initializer())

            # Extracting the running variables for operating the metrics
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="decoder/accuracy")
            print("running vars are ", running_vars)
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            sess.run(running_vars_initializer)
            
            for epoch_idx in range(0, config["num_epochs"]):        
                # Training epoch
                self.single_epoch_execution(
                    model=model, session=sess, dataset=self.train_set, epoch_type="Train",
                    log_writer=log_writer_train, running_vars_initializer=running_vars_initializer,
                    display_freq=TrainingManager.DISPLAY_FREQ)

                if model.global_step.eval(session=sess) % TrainingManager.VALIDATE_FREQ == 0:
                    avg_loss, avg_teacher_forcing_accuracy = self.single_epoch_execution(
                        model=model, session=sess, dataset=self.valid_set, epoch_type="Validation",
                        log_writer=log_writer_val, running_vars_initializer=running_vars_initializer,
                        display_freq=TrainingManager.DISPLAY_FREQ)
                    print(" Get avg loss of ", avg_loss)
                    if self.should_early_stop(cost_score=avg_loss):
                        print("An early stop condition triggered => early stopping...")
                        break
                    else:
                        print("Should continue training - best valid metric is %s, avg_loss is %s" % (
                            best_valid_metric, avg_loss))
                        if avg_loss < best_valid_metric:
                            best_valid_metric = avg_loss
                            print("Got a new best valid metric of ", best_valid_metric)
                            if config["saving_last_model"]:
                                print('Saving the last model.. at ', config['model_saved_path'])
                                model.save(sess, config['model_saved_path'], global_step=model.global_step)

                # Increase the epoch index of the model
                model.global_epoch_step_op.eval()
                print('Epoch {0:} DONE'.format(model.global_epoch_step.eval(session=sess)))

            print(" Best average validation metric = %s" % best_valid_metric)
        
        # Return average validation metric as the outcome
        return best_valid_metric

    def seq2seq_execution(self, sess, decoding_model, batch_size, input_text):
        """ A single inference execution of a decoding model

        Arguments:
            sess: Tensorflow session to be used (connected with the decoding model)
            decoding_model: a CharacterSeq2seq model to be used
            batch_size: a provided batch size (normally should be 1)
            input_text: an input text to execute the seq2seq operation according to the
        decoding model

        Returns:
            output text (destination text according to the decoding model)
        """

        # Convert input text to a sequence of numbers
        input_seq = self.lang_data_preprocessor.src_text_to_seq(input_text=input_text)

        # Prepare to feed into TF graph (as a batch of duplicated input)
        batch_of_input_seq = np.tile(input_seq, batch_size
                                     ).reshape([batch_size, -1])
        batch_of_length_seq = np.tile(len(input_text), batch_size)

        # Executing on the batch
        decoded_output = decoding_model.predict(
            sess,  encoder_inputs=batch_of_input_seq,
            encoder_inputs_length=batch_of_length_seq)
        # Extracting the first row of output
        first_row_out = decoded_output[0, :, :]

        # Convert the output back to a sequence of integers
        first_row_seq = first_row_out.T.flatten()

        # Convert the sequence of integers back to text
        output_text = self.lang_data_preprocessor.dest_seq_to_text(first_row_seq)
        return output_text
