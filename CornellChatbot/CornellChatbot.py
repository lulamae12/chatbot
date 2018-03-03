import pandas as pd #pandas datareader
import numpy as np
import tensorflow as tf
import re
import time
tf.__version__

#loads cornell data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

#sentences to train
lines[:10]
conv_lines[:10]

#dict to map line id with text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
#list of converstions line ids
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
convs[:10]

#sort into questions(input) or answers (target/output)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

#debugtest
limit = 0
for i in range(limit, limit+5):
    print(questions[i])
    print(answers[i])
    print()

#compare question/answer langth
print(f"Question Length:",len(questions))
print(f"Answer Length:",len(answers))

def textcleanup(text):
#cleansup text by removing unnecessary characters like is and reformats.
    #Returns the string obtained by replacing the leftmost non-overlapping occurrences of pattern in the string
    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text
#cleaned up questions
clean_questions = []
for question in questions:
    clean_questions.append(textcleanup(question))
#cleaned up answers
clean_answers = []
for answer in answers:
    clean_answers.append(textcleanup(answer))

#more clean debug prints.
limit = 0
for i in range(limit, limit+5):
    print(clean_questions[i])
    print(clean_answers[i])
    print()

# Find the length of sentences
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))

# Creates a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])
lengths.describe()
print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))

#cleans up querys that are to long/short
min_line_length = 2
max_line_length = 20

short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1
print("Number of questions:", len(short_questions))
print("Number of answers:", len(short_answers))
print("% of total data used: {}%".format(round(len(short_questions)/len(questions),4)*100))


#dictionary of frequently used vocab words
vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

# Remove rare words from the vocabulary.
#aims to replace fewer than 5% of words with <UNK>
threshold = 10
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1

print("Size of total vocab:", len(vocab))
print("Size of vocab that will be used:", count)

# In case you want to use a different vocabulary size for the source and target text,
# we can set different threshold values.
# however, dictionaries will still be created to provide a unique integer for each word.
questions_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1

answers_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += 1

#custom tokens for vocab dictionaries
# PAD = FILLER
# EOS = END OF STRING
# UNK = UNKNOWN/WORD NOT IN VOCABUALRY
# GO = START DECODING
#padding/go/eos example
#Q:"how are you?"
#A:"i am fine."
#Q : [ PAD, PAD, PAD, PAD, PAD, PAD, “?”, “you”, “are”, “How” ]
#A : [ GO, “I”, “am”, “fine”, “.”, EOS, PAD, PAD, PAD, PAD ]
#


codes = ['<PAD>','<EOS>','<UNK>','<GO>']

for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int)+1
for code in codes:
    answers_vocab_to_int[code] = len(answers_vocab_to_int)+1

#creat a dictinare to map unique ints to their corresponding words
questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}
#debug check length
print(len(questions_vocab_to_int))
print(len(questions_int_to_vocab))
print(len(answers_vocab_to_int))
print(len(answers_int_to_vocab))

#adds token at the end of answer to end it
for i in range(len(short_answers)):
    short_answers[i] += ' <EOS>'

#change text to integer
#replace unknown words in the vocabualry with UNK
questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_int.append(ints)
answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    answers_int.append(ints)

#debug to check the lengths
print(len(questions_int))
print(len(answers_int))

#calculation to determine what percentage of words are unknown and replaced with unk
word_count = 0
unk_count = 0

for question in questions_int:
    for word in question:
        if word == questions_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
for answer in answers_int:
    for word in answer:
        if word == answers_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1

unk_ratio = round(unk_count/word_count,4)*100
print("Total number of words:", word_count)
print("Number of times <UNK> is used:", unk_count)
print("Percent of words that are <UNK>: {}%".format(round(unk_ratio,3)))

#answer/question sorting section begins here:
#they will be sorted by length
#this is done to redice the amount of padding when training with datasets
#this is done in an effort to speed up the training process
sorted_questions = []
sorted_answers = []

for length in range(1, max_line_length+1):
    for i in enumerate(questions_int):
        if len(i[1]) == length:
            sorted_questions.append(questions_int[i[0]])
            sorted_answers.append(answers_int[i[0]])
print(f"Sorted question length:",len(sorted_questions))
print(f"Sorted answer length:",len(sorted_answers),'\n')
for i in range(3):
    print(sorted_questions[i])
    print(sorted_answers[i])
    print()


#begin Tensorflow
#for future refrence, the tensorflow python api is located here:https://www.tensorflow.org/api_docs/python/

#A tensor is a generalization of vectors and matrices to potentially higher dimensions.
#Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.

#nonverbose definition of a tensor: tensors are arrays of numbers or functions that can transform based on certain rules.
def model_inputs():
    '''Create placeholders for inputs to the model'''
    input_data = tf.placeholder(tf.int32, [None,None], name='input')#tf.placeholder Inserts a placeholder for a tensor that will be always fed.
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')#learning_rate variable is created
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')#keep_prob is made to keep probabilty

    return input_data, targets, lr, keep_prob

def process_encoding_input(target_data, vocab_to_int, batch_size):#converts input so tensor can read it
    '''Remove the last word id from each batch and concatanate the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0,0], [batch_size, -1], [1, 1])#loops starting at 0,0 and then goes to 1,1
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):#rnn is a recurrent neural network. applies filters for recognition.
    '''Creates the encoding layer'''
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)#Long Short Term Memory networks or “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)#cells act kind of like a conveyor becasue they help keep the network moving through LSTMSs
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,#cell forward, moves forward a cell in the graph
                                                   cell_bw = enc_cell,#cell backward, moves back a cell in the graph
                                                   sequence_length = sequence_length,
                                                   inputs = rnn_inputs,
                                                   dtype=tf.float32)
    return enc_state


#trains decoing layer
#the attention mechanism is the way to pay attention to certain parts of data so it dosent have to rely on a single thing. this allows the decoder to select any part of the sentence sepednming on the situation.
#there are 2 diffrent types of seq2seq attention models for tensorflow, luong attention and bahdanau attention, whihic is the one that is being used in this program
#this program uses bahdanau becasue it can read layers that are hidden and it can read them forward or backward which gives it a better chance at interpreting text and responding with the appropriate text correctlly.

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                        output_fn, keep_prob, batch_size):
    '''decodes the training data'''

    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])#tfzeros creates a tensor with all elements set to 0

    att_keys, att_vals, att_score_fn, att_construct_fn = \
            tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",#see top of def
                                                 num_units=dec_cell.output_size)

    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],#seq2seq is a module for dyanmic decoding
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name = "attn_dec_train")
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                              train_decoder_fn,
                                                              dec_embed_input,
                                                              sequence_length,
                                                              scope=decoding_scope)
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    return output_fn(train_pred_drop)
#logits are a function that infer probabilties
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):
    '''Decodes the prediction data'''

    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = \
            tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",#see top of above def
                                                 num_units=dec_cell.output_size)

    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn,
                                                                         encoder_state[0],
                                                                         att_keys,
                                                                         att_vals,
                                                                         att_score_fn,
                                                                         att_construct_fn,
                                                                         dec_embeddings,
                                                                         start_of_sequence_id,
                                                                         end_of_sequence_id,
                                                                         maximum_length,
                                                                         vocab_size,
                                                                         name = "attn_dec_inf")
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                                infer_decoder_fn,
                                                                scope=decoding_scope)

    return infer_logits
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size):
    '''Creates the decoding cell and input the parameters for the training and inference decoding layers'''

    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        weights = tf.truncated_normal_initializer(stddev=0.1)#weight refers to the strength or amplitude of a connection between two nodes or "tensors" in this case
        biases = tf.zeros_initializer()#pretty self explanotory. allows the program to lean to one side of two options based on parameters
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, #inpython lambda allows creation of anon functions.
                                                                vocab_size,
                                                                None,
                                                                scope=decoding_scope,
                                                                weights_initializer = weights,
                                                                biases_initializer = biases)

        train_logits = decoding_layer_train(encoder_state,
                                            dec_cell,
                                            dec_embed_input,
                                            sequence_length,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state,
                                            dec_cell,
                                            dec_embeddings,
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'],
                                            sequence_length - 1,
                                            vocab_size,
                                            decoding_scope,
                                            output_fn, keep_prob,
                                            batch_size)

    return train_logits, infer_logits
    #very simialr to previous
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size,
                  questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers,
                  questions_vocab_to_int):

    '''Use the previous functions to create the training and inference logits'''

    enc_embed_input = tf.contrib.layers.embed_sequence(input_data,
                                                       answers_vocab_size+1,
                                                       enc_embedding_size,
                                                       initializer = tf.random_uniform_initializer(0,1))
    #encoding state based on layer
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size+1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    train_logits, infer_logits = decoding_layer(dec_embed_input,
                                                dec_embeddings,
                                                enc_state,
                                                questions_vocab_size,
                                                sequence_length,
                                                rnn_size,
                                                num_layers,
                                                questions_vocab_to_int,
                                                keep_prob,
                                                batch_size)
    return train_logits, infer_logits
