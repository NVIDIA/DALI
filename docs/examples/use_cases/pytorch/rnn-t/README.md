# DISCLAIMER
This codebase is a work in progress. There are known and unknown bugs in the implementation, and has not been optimized in any way.

MLPerf has neither finalized on a decision to add a speech recognition benchmark, nor as this implementationn/architecture as a reference implementation.

# 1. Problem 
Speech recognition accepts raw audio samples and produces a corresponding text transcription.

# 2. Directions
See https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/README.md. This implementation shares significant code with that repository.

# 3. Dataset/Environment
### Publication/Attribution
["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) provides over 1000 hours of speech data in the form of raw audio.
### Data preprocessing
What preprocessing is done to the the dataset? 
### Training and test data separation
How is the test set extracted?
### Training data order
In what order is the training data traversed?
### Test data order
In what order is the test data traversed?
### Simulation environment (RL models only)
Describe simulation environment briefly, if applicable. 
# 4. Model
### Publication/Attribution
Cite paper describing model plus any additional attribution requested by code authors 
### List of layers 
Brief summary of structure of model
### Weight and bias initialization
How are weights and biases initialized
### Loss function
Transducer Loss
### Optimizer
TBD, currently Adam
# 5. Quality
### Quality metric
Word Error Rate (WER) across all words in the output text of all samples in the validation set.
### Quality target
What is the numeric quality target
### Evaluation frequency
TBD
### Evaluation thoroughness
TBD