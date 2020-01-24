---
title: "Writer's Block? RNNs can help!"
linktitle: "Writer's Block? RNNs can help!"

date: "2019-10-23T17:30:00"
lastmod: "2019-10-23T17:30:00"

draft: false
toc: true
type: docs

weight: 6

menu:
  core_fa19:
    parent: Fall 2019
    weight: 6

authors: ["brandons209", "ionlights", ]

urls:
  youtube: ""
  slides:  ""
  github:  "https://github.com/ucfai/core/blob/master/fa19/2019-10-23-rnns/2019-10-23-rnns.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/core-fa19-rnns"
  colab:   "https://colab.research.google.com/github/ucfai/core/blob/master/fa19/2019-10-23-rnns/2019-10-23-rnns.ipynb"

room: "MSB 359"
cover: "https://i.imgur.com/EIt4Ilr.png"

categories: ["fa19"]
tags: ["neural-nets", "recurrent-nets", "LSTMs", "Embeddings", ]
description: >-
  This lecture is all about Recurrent Neural Networks. These are networks with memory, which means they can learn from sequential data such as speech, text, videos, and more. Different types of RNNs and strategies for building them will also be covered. The project will be building a LSTM-RNN to generate new original scripts for the TV series “The Simpsons”. Come and find out if our networks can become better writers for the show!
---
```python
# This is a bit of code to make things work on Kaggle
import os
from pathlib import Path

if os.path.exists("/kaggle/input/ucfai-core-fa19-rnns"):
    DATA_DIR = Path("/kaggle/input/ucfai-core-fa19-rnns")
else:
    DATA_DIR = Path("data/")
```

# Generate new Simpson scripts with LSTM RNN
## Link to slides [here](https://docs.google.com/presentation/d/1ztu3_4xsuWH1FAsGqnGJkP_TBKS3Q27FPUMFkraMB34/edit?usp=sharing)
In this project, we will be using an LSTM with the help of an Embedding layer to train our network on an episode from the Simpsons, specifically the episode "Moe's Tavern". This is taken from [this](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) dataset on kaggle. This model can be applied to any text. We could use more episodes from the Simpsons, a book, articles, wikipedia, etc. It will learn the semantic word associations and being able to generate text in relation to what it is trained on.

First, lets import all of our libraries we need.

```python
# general imports
import numpy as np
import time
import os
import pickle
import glob

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

# tensorboardX
from tensorboardX import SummaryWriter
```

#### The cell below contains a bunch of helper functions for us to use today, dealing with string manipulation and printing out epoch results. Feel free to take a look after the workshop! 

```python
"""
Loads text from specified path, exits program if the file is not found.
"""
def load_script(path):

    if not os.path.isfile(path):
        print("Error! {} was not found.".format(path))
        sys.exit(1)

    with open(path, 'r') as file:
        text = file.read()
    return text
 
# saves dictionary to file for use later
def save_dict(dict, filename):
    dir = 'data/dictionaries/' + filename
    with open(dir, 'wb') as file:
        pickle.dump(dict, file)

# loads dictionary from file
def load_dict(filename):
    dir = 'data/dictionaries/' + filename
    with open(dir, 'rb') as file:
         dict = pickle.load(file)
    return dict

#dictionaries for tokenizing puncuation and converting it back
punctuation_to_tokens = {'!':' ||exclaimation_mark|| ', ',':' ||comma|| ', '"':' ||quotation_mark|| ',
                          ';':' ||semicolon|| ', '.':' ||period|| ', '?':' ||question_mark|| ', '(':' ||left_parentheses|| ',
                          ')':' ||right_parentheses|| ', '--':' ||dash|| ', '\n':' ||return|| ', ':':' ||colon|| '}

tokens_to_punctuation = {token.strip(): punc for punc, token in punctuation_to_tokens.items()}

#for all of the puncuation in replace_list, convert it to tokens
def tokenize_punctuation(text):
    replace_list = ['.', ',', '!', '"', ';', '?', '(', ')', '--', '\n', ':']
    for char in replace_list:
        text = text.replace(char, punctuation_to_tokens[char])
    return text

#convert tokens back to puncuation
def untokenize_punctuation(text):
    replace_list = ['||period||', '||comma||', '||exclaimation_mark||', '||quotation_mark||',
                    '||semicolon||', '||question_mark||', '||left_parentheses||', '||right_parentheses||',
                    '||dash||', '||return||', '||colon||']
    for char in replace_list:
        if char == '||left_parentheses||':#added this since left parentheses had an extra space
            text = text.replace(' ' + char + ' ', tokens_to_punctuation[char])
        text = text.replace(' ' + char, tokens_to_punctuation[char])
    return text

"""
Takes text already converted to ints and a sequence length and returns the text split into seq_length sequences and generates targets for those sequences
"""
def gen_sequences(int_text, seq_length):
    seq_text = []
    targets = []
    for i in range(0, len(int_text) - seq_length, 1):
        seq_in = int_text[i:i + seq_length]
        seq_out = int_text[i + seq_length]
        seq_text.append([word for word in seq_in])
        targets.append(seq_out)#target is next word after the sequence
    return np.array(seq_text, dtype=np.int32), np.array(targets, dtype=np.int32)

```

```python
from tabulate import tabulate

BATCH_TEMPLATE = "Epoch [{} / {}], Batch [{} / {}]:"
EPOCH_TEMPLATE = "Epoch [{} / {}]:"
TEST_TEMPLATE = "Epoch [{}] Test:"

def print_iter(curr_epoch=None, epochs=None, batch_i=None, num_batches=None, writer=None, msg=False, **kwargs):
    """
    Formats an iteration. kwargs should be a variable amount of metrics=vals
    Optional Arguments:
        curr_epoch(int): current epoch number (should be in range [0, epochs - 1])
        epochs(int): total number of epochs
        batch_i(int): current batch iteration
        num_batches(int): total number of batches
        writer(SummaryWriter): tensorboardX summary writer object
        msg(bool): if true, doesn't print but returns the message string

    if curr_epoch and epochs is defined, will format end of epoch iteration
    if batch_i and num_batches is also defined, will define a batch iteration
    if curr_epoch is only defined, defines a validation (testing) iteration
    if none of these are defined, defines a single testing iteration
    if writer is not defined, metrics are not saved to tensorboard
    """
    if curr_epoch is not None:
        if batch_i is not None and num_batches is not None and epochs is not None:
            out = BATCH_TEMPLATE.format(curr_epoch + 1, epochs, batch_i, num_batches)
        elif epochs is not None:
            out = EPOCH_TEMPLATE.format(curr_epoch + 1, epochs)
        else:
            out = TEST_TEMPLATE.format(curr_epoch + 1)
    else:
        out = "Testing Results:"

    floatfmt = []
    for metric, val in kwargs.items():
        if "loss" in metric or "recall" in metric or "alarm" in metric or "prec" in metric:
            floatfmt.append(".4f")
        elif "accuracy" in metric or "acc" in metric:
            floatfmt.append(".2f")
        else:
            floatfmt.append(".6f")

        if writer and curr_epoch:
            writer.add_scalar(metric, val, curr_epoch)
        elif writer and batch_i:
            writer.add_scalar(metric, val, batch_i * (curr_epoch + 1))

    out += "\n" + tabulate(kwargs.items(), headers=["Metric", "Value"], tablefmt='github', floatfmt=floatfmt)

    if msg:
        return out
    print(out)
```

## Dataset statistics
Before starting our project, we should take a look at the data we are dealing with. We are loading in a single episode from the Simpsons, but you can load in any other text from a `.txt` file. There is also an included Trump's Tweets dataset and a loop if you want to add multiple text files in at once.

```python
script_text = load_script(str(DATA_DIR / 'moes_tavern_lines.txt'))
#script_text = load_script(str(DATA_DIR / 'harry-potter.txt'))
# if you want to load in your own data, add it a directory called data (as many text files as you want)
# and uncomment this here: (remember that these stats wont be accurate unless you use the simpsons dataset)
# spript_text = ""
#for script in sort(glob.glob(str(DATA_DIR))):
#    script_text += load_script(script)

print('----------Dataset Stats-----------')
print('Approximate number of unique words: {}'.format(len({word: None for word in script_text.split()})))
scenes = script_text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {:.0f}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {:.0f}'.format(np.average(word_count_sentence)))
```

## Tokenize Text
In order to prepare our data for our network, we need to tokenize the words. That is, we will be converting every unique word and punctuation into an integer. Before we do that, we need to make the punctuation easier to convert to a number. For example, we will be taking any new lines and converting them to the word "||return||". This makes the text easier to tokenize and pass into our model. The functions that do this are in the helper functions code block above.

A note on tokenizing: 0 is a reserved integer, that is its not used to represent any words. So our integers for our words will start at 1. This is needed as when we use the model to generate new text, it needs a starting point, known as a seed. If this seed is smaller than our sequence length, then the function pad_sequences will pad that seed with 0's in order to represent "nothing". This help reduces noise in the network. Think of it as whitespace, it doesn't change the meaning to the input phrase.

This is the list of punctuation and special characters that are converted, notice that spaces are put before and after to make splitting the text easier:
- '!' : ' ||exclaimation_mark|| '
- ',' : ' ||comma|| '
- '"' : ' ||quotation_mark|| '
- ';' : ' ||semicolon|| '
- '.' : ' ||period|| '
- '?' : ' ||question_mark|| '
- '(' : ' ||left_parentheses|| '
- ')' : ' ||right_parentheses|| '
- '--' : ' ||dash|| '
- '\n' : ' ||return|| '
- ':' : ' ||colon|| '

We also convert all of the text to lowercase as this reduces the vocabulary list and trains the network faster.

```python
script_text = tokenize_punctuation(script_text) # helper function to convert non-word characters
script_text = script_text.lower()

script_text = script_text.split() # splits the text based on spaces into a list
```

## Creating Conversion Dictionaries and Input Data
Now that the tokens have been generated, we will create some dictionaries to convert our tokenized integers back to words, and words to integers. We will also generate our inputs and targets to pass into our model. 

To do this, we need to specify the sequence length, which is the amount of words we pass into the model at one time. I choose 12 for the average sentence length seen in Dataset Stats, but feel free to change this. A sequence length of 1 is just one word, so we could get better output depending on our sequence length. We use the helper function gen_sequences to do this for us.

The dataset and dataloader is defined using the specified batch_size.

The targets are simply just the next word in our text. So if we have a sentence: "Hi, how are you?" and we input "Hi, how are you" our target for this sentence will be "?".

```python
sequence_length = 12
batch_size = 64

int_to_word = {i+1: word for i, word in enumerate(set(script_text))}
word_to_int = {word: i for i, word in int_to_word.items()} # flip word_to_int dict to get int to word
int_script_text = np.array([word_to_int[word] for word in script_text], dtype=np.int32) # convert text to integers
int_script_text, targets = gen_sequences(int_script_text, sequence_length) # transform int_script_text to sequences of sequence_length and generate targets

vocab_size = len(word_to_int) + 1 # add one since indexes are 1 to length
# convert to tensors and define dataset
dataset = TensorDataset(torch.from_numpy(int_script_text), torch.from_numpy(targets))
# define dataloader for the dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("Number of vocabulary: {}, Dataloader size: {}".format(vocab_size, len(dataloader)))
```

## Building the Model
Here is the fun part, building our model. We will use LSTM cells and an Embedding layer, with a fully connected Linear layer at the end for the prediction. Documentation for LSTM cells can be found [here](https://pytorch.org/docs/stable/nn.html#lstm) and for embedding [here](https://pytorch.org/docs/stable/nn.html#embedding).

An LSTM network can be defined simply as:    
```
nn.LSTM(input_size, hidden_size, num_layers, dropout, batch_first=True)
```
- dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
           Introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
- hidden_size: Number of LSTM cells in each layer
- batch_first: Whether the first dimensions is batch_size or sequence_length. Leave this to true for our model, as batch_size is first.

Import to note is that the output of the LSTM network here has the shape of (seq_len, batch_size, hidden_size), so in our forward pass you need to use the `transpose` tensor method to swap the batch_size and seq_len axes before input into the Linear layer. Also, for input into the Linear layer from the LSTM layer requires the last step of the output. Remember, the LSTM network returns the output for **every state**, so make sure to get the last one.

An embedding layer can be defined as:
```
Embedding(input_dim, embed_size)
```
Our input dimension will be the length of our vocabulary, the size can be whatever you want to set it at, my case I used 300.
Our model will predict the next word based in the input sequence. We could also predict the next two words, or predict entire sentences. For now though we will just stick with one word.

```python
class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size=400, num_layers=1, dropout=0.3):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # batch_size is first
        self.hidden_dim = lstm_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        
        self.LSTM = nn.LSTM(input_size=embed_size, hidden_size=lstm_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.classifier = nn.Linear(lstm_size, vocab_size)
        
    def forward(self, x, prev_hidden):
        batch_size = x.size(0)
        out = self.embedding(x)
        out, hidden = self.LSTM(out, prev_hidden)
        # the output from the LSTM needs to be flattened for the classifier, so reshape output to: (batch_size * seq_len, hidden_dim)
        out = out.contiguous().view(-1, self.hidden_dim)
        
        out = self.classifier(out)
        
        # reshape to split apart the batch_size * seq_len dimension
        out = out.view(batch_size, -1, self.vocab_size)
        
        # only need the output of the layer, so remove the middle seq_len dimension
        out = out[:, -1]
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
    
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
 
        
        return hidden
```

## Hyperparameters and Compiling the Model
The Adam optimizer is very effective and has built in dynamic reduction of the learning rate, so let's use that. We will also set the learning rate, epochs, and batch size.

We use the CrossEntropyLoss, which requires raw logits as input, since softmax is built into the loss function.

Dropout should be a bit high as we are training on a small amount of data, so our model is prone to overfit quickly.

```python
### BEGIN SOLUTION
model = LSTM_Model(vocab_size, 300, lstm_size=400, num_layers=2, dropout=0.5)
### END SOLUTION
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: {}".format(device))
model.to(device)

learn_rate = 0.001

# write out the optimizer and criterion here, using CrossEntropyLoss and the Adam optimizer
### BEGIN SOLUTION
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
criterion = nn.CrossEntropyLoss()
### END SOLUTION

# torch summary has a bug where it won't work with embedding layers
print(model)

if device == 'cuda':
    # helps with runtime of model, use if you have a constant batch size
    cudnn.benchmark = True
```

```python
# load weights if continuing training
load = torch.load("best.weights.pt")
model.load_state_dict(load["net"])
print("Loaded model from epoch {}.".format(load["epoch"]))
```

## Training
Now it is time to train the model. We will use tensorboardX to graph our loss and we will save the checkpoint everytime our training loss decreases. We do not use validation data because we want the model to be closely related to how our text is constructed.

The Tensorboard is commented out right now, since we are running on kaggle. If you run locally check you can uncomment the tensorbardX lines and view the tensorboard by:
- Installing tensorboard `pip install tensorboard`
- Running tensorboard --logdir=tensorboard_logs in a terminal
- Going to the link it gives you.

Note that the model overfits easily since we don't have much data, so train for a small number of epochs.

```python
epochs = 7

# view tensorboard with command: tensorboard --logdir=tensorboard_logs
# os.makedirs("tensorboard_logs", exist_ok=True)
# os.makedirs("checkpoints", exist_ok=True)

# ten_board = SummaryWriter('tensorboard_logs/run_{}'.format(start_time))
weight_save_path = 'best.weights.pt'
print_step = len(dataloader) // 20
model.train()
best_loss = 0
start_time = time.time()

for e in range(epochs):
    train_loss = 0
    
    # get inital hidden state
    hidden = model.init_hidden(batch_size)
    hidden = (hidden[0].to(device), hidden[1].to(device))
    
    for i, data in enumerate(dataloader):
        # make sure you iterate over completely full batches, only
        if len(data[0]) < batch_size:
            break
            
        inputs, targets = data
        inputs, targets = inputs.type(torch.LongTensor).to(device), targets.type(torch.LongTensor).to(device)
        
        hidden = tuple([each.data for each in hidden])
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        
        if i % print_step == 0:
            print_iter(curr_epoch=e, epochs=epochs, batch_i=i, num_batches=len(dataloader), loss=train_loss/(i+1))
    
    # print iteration takes the tensorboardX writer and adds the metrics we have to it
    # print_iter(curr_epoch=e, epochs=epochs, writer=writer, loss=train_loss/len(train_dataloader))
    print_iter(curr_epoch=e, epochs=epochs, loss=train_loss/len(dataloader))
    
    if e == 0:
        best_loss = train_loss
    elif train_loss < best_loss:
        print('\nSaving Checkpoint..\n')
        state = {'net': model.state_dict(), 'loss': train_loss, 'epoch': e, 'sequence_length': sequence_length, 'batch_size': batch_size, 'int_to_word': int_to_word, 'word_to_int': word_to_int}
        torch.save(state, weight_save_path)
        best_loss = train_loss

print("Model took {:.2f} minutes to train.".format((time.time() - start_time) / 60))
```

## Testing the Model
Testing the model simply requires that we convert the output integer back into a word and build our generated text, starting from a seed we define. However, we might get better results by instead of doing an argmax to find the highest probability of what the next word should be, we can take a sample of the top possible words and choose one from there. 

This is done by taking a "temperature" which defines how many predictions we will consider as the next possible word. A lower temperature means the word picked will be closer to the word with the highest probability. Then using a random selection to choose a word. Try it with using both. Setting a temperature of 0 will just use argmax on the entire prediction.

```python
#load model if returning to this notebook for testing, model that I trained:
load = torch.load(weight_save_path)
model.load_state_dict(load["net"])
```

```python
model.eval()

def sample(prediction, temp=0):
    if temp <= 0:
        return np.argmax(prediction)
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temp
    expo_prediction = np.exp(prediction)
    prediction = expo_prediction / np.sum(expo_prediction)
    probabilities = np.random.multinomial(1, prediction, 1)
    return np.argmax(probabilities)

def pad_sequences(sequence, maxlen, value=0):
    while len(sequence) < maxlen:
        sequence = np.insert(sequence, 0, value)
    if len(sequence) > maxlen:
        sequence = sequence[len(sequence) - maxlen:]
    return sequence

#generate new script
def generate_text(seed_text, num_words, temp=0):
    input_text= seed_text
    for _  in range(num_words):
        #tokenize text to ints
        int_text = tokenize_punctuation(input_text)
        int_text = int_text.lower()
        int_text = int_text.split()
        int_text = np.array([word_to_int[word] for word in int_text], dtype=np.int32)
        #pad text if it is too short, pads with zeros at beginning of text, so shouldnt have too much noise added
        int_text = pad_sequences(int_text, maxlen=sequence_length)
        int_text = np.expand_dims(int_text, axis=0)
        # init hiddens state
        hidden = model.init_hidden(int_text.shape[0])
        hidden = (hidden[0].to(device), hidden[1].to(device))
        #predict next word:
        prediction, _ = model(torch.from_numpy(int_text).type(torch.LongTensor).to(device), hidden)
        prediction = prediction.to("cpu").detach()
        prediction = F.softmax(prediction, dim=1).data
        prediction = prediction.numpy().squeeze()
        output_word = int_to_word[sample(prediction, temp=temp)]
        #append to the result
        input_text += ' ' + output_word
    #convert tokenized punctuation and other characters back
    result = untokenize_punctuation(input_text)
    return result
```

```python
#input amount of words to generate, and the seed text, good options are 'Homer_Simpson:', 'Bart_Simpson:', 'Moe_Szyslak:', or other character's names.:
seed = 'Homer_Simpson:'
num_words = 200
temp = 0.5

# print amount of characters specified.
print("Starting seed is: {}\n\n".format(seed))
print(generate_text(seed, num_words, temp=temp))
```

## Closing Thoughts
Remember that this model can be applied to any type of text, even code! So go and try different texts, like the (not) included Harry Potter book. (for time purposes, I would not use the whole book, as training would take a long time.)

Try different hyperparameters and model sizes, as you can get some better results.
