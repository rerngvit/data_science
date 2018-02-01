import numpy as np


class DataFeedIterator:
    """An iterator class to represent the datafeed for the Tensorflow graph"""

    def __init__(self, lang_preprocessor, 
                 src_texts, dest_texts, batch_size):
        """

        :param lang_preprocessor: a corresponding CharacterLevelSeq2SeqPreprocessor object
        :param src_texts: list of source strings to create for this iterator
        :param dest_texts: list of destination strings to create for this iterator
        (same number of rows as src_texts)
        :param batch_size: the batch size to be used during the feeding process
        """

        self.lang_preprocessor = lang_preprocessor
        self.src_texts = src_texts  # [num_rows x 1]
        self.dest_texts = dest_texts  # [num_rows x 1]
        self.batch_size = batch_size
        self.num_rows = len(src_texts)  # number of rows
        assert(self.num_rows == len(dest_texts))  # it should be exactly equal
        self.data_pointer = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_rows // self.batch_size
    
    def reset(self):
        self.data_pointer = 0
    
    def __next__(self): 
        if self.data_pointer + self.batch_size >= self.num_rows:
            self.reset()
            raise StopIteration
        chunk_src_texts = self.src_texts[self.data_pointer:(self.data_pointer + self.batch_size)]
        chunk_dest_texts = self.dest_texts[self.data_pointer:(self.data_pointer + self.batch_size)]
        chunk_src_data_seq = np.vstack([self.lang_preprocessor.src_text_to_seq(chunk_text) 
                                        for chunk_text in chunk_src_texts])
        chunk_dest_data_seq = np.vstack([self.lang_preprocessor.dest_text_to_seq(chunk_text) 
                                        for chunk_text in chunk_dest_texts])               
        lengths_src = np.array([len(s) for s in chunk_src_texts])
        lengths_dest = np.array([len(s) for s in chunk_dest_texts])
        
        self.data_pointer += self.batch_size
        
        return chunk_src_data_seq, lengths_src, chunk_dest_data_seq, lengths_dest
