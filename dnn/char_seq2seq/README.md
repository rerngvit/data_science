# Character-level Seq2Seq model with attention
This repository contains implementation for a character-level seq2seq (with attention) model in Tensorflow. The code consists of an example notebook and supporting python files. This notebook demonstrates a working example of applying the character-level sequence-to-sequence with attention model to address a language correction problem. We limit the scope of the problem to fixing a single missing article (a, an, the). This means that given an input text with an article missing, the model would recommend a language-corrected text as an output.

In particular, the folder structure is the following.
- preprocessor.py - containing the character-level preprocessor code responsible for data preprocessing, i.e., converting from raw data to the format that is ready to be trained by a Tensorflow graph for recurrent network modelling.
- data_iterator.py - helping code for iterating batches of dataset for feeding into the training operation
- tf_graph.py - the Tensorflow modeling code
- trainining_manager.py - the training management code, which is responsible for data splitting, executing the training, and controling the workflow of the training operation
- example-missing-articles-correction.ipynb - the key example notebook for the language correction application for this model.
