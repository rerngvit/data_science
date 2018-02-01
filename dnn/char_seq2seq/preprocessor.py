import numpy as np
import warnings
import time
import os
import json


class CharacterLevelSeq2SeqPreprocessor:
    """ Class representing character-level preprocessors.

    This class converts from raw string data to format that is ready to be fed
    to a TF graph.

    """

    def __init__(self, src_data_path, dest_data_path,
                 max_number_of_samples=-1,
                 max_character_length=50):
        """
        Arguments:
            src_data_path: the one-data-per-line file path of the source text input
            dest_data_path: the one-data-per-line file path of the destination text input
            max_number_of_samples: the maximum number of rows to use (from the beginning of the files)
            MAX_CHARACTER_LENGTH: the maximum number of characters to use. The samples that have
                higher number of characters than this will be ignored. This is needed to make sure
                that the GPU memory is available for training.
        """

        self.MAX_NUMBER_OF_SAMPLES = max_number_of_samples
        self.MAX_CHARACTER_LENGTH = max_character_length
        self.PADDED_character = "\t"
        self.EOS_character = "\n"
        self.SEQ_START_CHARACTER = "\r"
        self.special_characters = [self.PADDED_character, self.EOS_character, self.SEQ_START_CHARACTER]
    
        # Passing through the data to understand existing tokens
        self.src_tokens = set()
        self.dest_tokens = set()
        src_lines = open(src_data_path, encoding='utf-8')
        dest_lines = open(dest_data_path, encoding='utf-8')
        
        num_samples = 0
        self.max_src_seq_length, self.max_dest_seq_length = 0, 0        
        for src, dest in zip(src_lines, dest_lines):
            if len(src) <= self.MAX_CHARACTER_LENGTH and len(dest) <= self.MAX_CHARACTER_LENGTH:
                src = self.preprocess(src) + self.EOS_character
                dest = self.SEQ_START_CHARACTER + self.preprocess(dest) + self.EOS_character
                for src_token in src:
                    if src_token not in self.src_tokens:
                        self.src_tokens.add(src_token)
                for dest_token in dest:
                    if dest_token not in self.dest_tokens:
                        self.dest_tokens.add(dest_token)
                        
                if len(src) > self.max_src_seq_length:
                    self.max_src_seq_length = len(src)
                
                if len(dest) > self.max_dest_seq_length:
                    self.max_dest_seq_length = len(dest)
                    
                num_samples += 1
                if self.MAX_NUMBER_OF_SAMPLES != -1 and \
                    num_samples >= self.MAX_NUMBER_OF_SAMPLES:
                    break
        
        print(" Processing in total %s lines" % num_samples)
        src_lines.close()
        dest_lines.close()

        # Add special characters into the tokens
        self.src_tokens.update(self.special_characters)
        self.dest_tokens.update(self.special_characters)
        self.src_tokens = sorted(list(self.src_tokens))
        self.dest_tokens = sorted(list(self.dest_tokens))
        self.num_src_tokens = len(self.src_tokens)
        self.num_dest_tokens = len(self.dest_tokens)
        
        # Create helping data structures
        # These are dictionary mapping between character and the character index
        self.src_token_index = dict([(char, i) for i, char in enumerate(self.src_tokens)])
        self.src_inv_token_index = dict([(i, char) for i, char in enumerate(self.src_tokens)])
        self.dest_token_index = dict([(char, i) for i, char in enumerate(self.dest_tokens)])
        self.dest_inv_token_index = dict([(i, char) for i, char in enumerate(self.dest_tokens)])

    def preprocess(self, text):
        """Preprocessing of raw input text by removing a sequence-start
        character, a padding character, as well as an ending-of-sentence
        character.

        Arguments:
             text: raw input text
        Returns:
            Preprocessed text
        """

        cur_text = text.replace(
            self.SEQ_START_CHARACTER, "").replace(
            self.PADDED_character, "").replace(self.EOS_character, "")
        return cur_text
    
    def __text_to_seq(self, input_text, max_seq_length, token_index):
        """Converting raw text to sequence of character indexes

        Arguments:
            input_text: raw input_text
            max_seq_length: the maximum number of character to consider for conversion
            token_index: the dictionary mapping from the token character to the token index
        Returns:
            Sequence of character indexes according to the input text
        """
        trimmed_input_text = input_text[:max_seq_length]
        
        input_seq = np.zeros(max_seq_length)  # create the data structure to retrieve input
        for t, char in enumerate(trimmed_input_text):
            input_seq[t] = token_index[char]
        return input_seq
    
    def src_text_to_seq(self, input_text):
        """Converting from raw source text to the sequence of characters

        Arguments:
            input_text: raw input source text
        Returns:
            Sequence of character indexes according to the raw input text
        """
        return self.__text_to_seq(input_text, self.max_src_seq_length,
                                  self.src_token_index)

    def dest_text_to_seq(self, input_text):
        """Converting from raw destination text to the sequence of characters

        Arguments:
            input_text: raw input destination text
        Returns:
            Sequence of character indexes according to the raw destination text
        """

        return self.__text_to_seq(input_text, self.max_dest_seq_length,
                                  self.dest_token_index)

    def __length_seq(self, input_seq, input_token_index):
        """Return distance from the beginning until the first EOS character

        Arguments:
            input_seq: sequence of character indexes
            input_token_index: dictionary mapping from character to index

        """
        # We need to extract the tuple + the first index that is found
        np_where_result = np.where(input_seq == input_token_index[
            self.EOS_character])[0]
        
        if len(np_where_result) > 0:
            return np_where_result[0] + 1
        else:
            return input_seq.shape[0]
    
    def __seq_to_text(self, data_seq, inv_token_index, input_token_index,
                      start_offset=0):
        """Convert from raw sequence index back to input text

        Arguments:
            data_seq: input sequence of character indexes
            inv_token_index: dictionary providing information from index -> character
            input_token_index: dictionary providing information from character -> index
            start_offset: the offset to start the conversion to raw text

        Returns:
            The converted raw text
        """
        # trim data_seq from the beginning until the encoder of EOS
        actual_length = self.__length_seq(data_seq, input_token_index)
        actual_data_seq = data_seq[:actual_length]  # cut until the length

        token_vecs = np.vectorize(lambda x: inv_token_index[x])(actual_data_seq)
        str_out = "".join(list(token_vecs[start_offset:]))
        return str_out
    
    def src_seq_to_text(self, data_seq):
        """ Convert from a sequence of src token ID representation to text """
        return self.__seq_to_text(data_seq, self.src_inv_token_index, 
                                  self.src_token_index, start_offset=0)
    
    def dest_seq_to_text(self, data_seq):
        """ Convert from a sequence of dest token ID representation to text """
        return self.__seq_to_text(data_seq, self.dest_inv_token_index, 
                                  self.dest_token_index, start_offset=0)

    def prepare_single_batch(self, input_text, batch_size):
        """ Duplicated a single text for the whole batch"""
        input_seq = self.src_text_to_seq(input_text=input_text)

        # Prepare to feed into TF graph (as a batch of duplicated input)
        batch_of_input_seq = np.tile(input_seq, batch_size
                                     ).reshape([batch_size, -1]).astype('int32')
        batch_of_length_seq = np.tile(len(input_text), batch_size).astype('int32')
        
        return batch_of_input_seq, batch_of_length_seq
