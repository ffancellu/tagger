#!/usr/bin/env python

import os
import numpy as np
import optparse
import codecs
from collections import OrderedDict
from utils import create_input
import loader

from data_processing import int_processor,ext_processor
from utils import models_path, evaluate_scope
from loader import word_mapping, char_mapping, tag_mapping
from loader import prepare_dataset_scope
from model import Model

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-P", "--PoS_tag", default="0",
    help="0 if None; 1 for normal and 2 for universal"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-v", "--pre_voc", default="",
    help="Location of the w2idx dict. for the pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-I", "--training_lang", default="en",
    help="Training lang (default: English)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model for testing purposes"
)
optparser.add_option(
    "-F", "--folder_name", default="system0",
    help="Name of the folder for the current experiment"
)

opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['pre_voc'] = opts.pre_voc
parameters['all_emb'] = opts.all_emb == 1
parameters['train_lang'] = opts.training_lang
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
parameters['folder_name'] = opts.folder_name

testing = opts.reload

# Add the POS tag info?
pos_tag = int(opts.PoS_tag)
parameters['pos_dim'] = opts.word_dim if pos_tag in [1,2] else 0

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
test_sets = opts.test.split(',')
for test_set in test_sets:
    assert os.path.isfile(test_set)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes','']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert pos_tag in [0,1,2]
# assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(parameters=parameters, models_path=models_path)
print "Model location: %s" % model.model_path

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']
tl = parameters['train_lang']

# Load data
train_set, valid_set, voc, dic_inv = int_processor.load_train_dev(
    True,
    False,
    tl,
    opts.train,
    opts.dev,
    models_path,
    "")
test_lex, test_tags, test_tags_uni, test_cue, _, test_y = int_processor.load_test(test_sets, voc, True, False, 'en', "")

train_lex, train_tags, train_tags_uni, train_cue, _, train_y = train_set
valid_lex, valid_tags, valid_tags_uni, valid_cue, _, valid_y = valid_set

dico_chars, char_to_id, id_to_char = char_mapping([[dic_inv['idxs2w'][t] for t in idx_sent] for idx_sent in train_lex])

# Index data
train_data = prepare_dataset_scope(
    [[dic_inv['idxs2w'][t] for t in idx_sent] for idx_sent in train_lex],
    train_lex,
    train_cue,
    train_tags_uni if pos_tag == 2 else train_tags,
    train_y,
    char_to_id)

dev_data = prepare_dataset_scope(
    [[dic_inv['idxs2w'][t] for t in idx_sent] for idx_sent in valid_lex],
    valid_lex,
    valid_cue,
    valid_tags_uni if pos_tag == 2 else valid_tags,
    valid_y,
    char_to_id)

test_data = prepare_dataset_scope(
    [[dic_inv['idxs2w'][t] for t in idx_sent] for idx_sent in test_lex],
    test_lex,
    test_cue,
    test_tags_uni if pos_tag == 2 else test_tags,
    test_y,
    char_to_id)

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

word_to_id = voc['w2idxs']

id_to_word = dic_inv['idxs2w']
id_to_tags = dic_inv['idxs2tuni'] if pos_tag == 2 else dic_inv['idxs2t']
id_to_tag = dic_inv['idxs2y']

# Save the mappings to disk
print 'Saving the mappings to disk...'
model.save_mappings(id_to_word, id_to_char, id_to_tag)

# Add n_pos to the parameters
parameters['n_pos'] = len(id_to_tags)

# Build the model
print parameters
f_train, f_eval = model.build(**parameters)

#
# Train network
#

if not testing:
    n_epochs = 50  # number of epochs over the training set
    freq_eval = 500  # evaluate on dev every freq_eval steps
    best_dev = -np.inf
    best_test = -np.inf
    count = 0
    for epoch in xrange(n_epochs):
        epoch_costs = []
        print "Starting epoch %i..." % epoch
        for i, index in enumerate(np.random.permutation(len(train_data))):
            count += 1
            input = create_input(train_data[index], parameters, True, False if pos_tag==0 else True)
            new_cost = f_train(*input)
            epoch_costs.append(new_cost)
            if i % 50 == 0 and i > 0 == 0:
                print "%i, cost average: %f" % (i, np.mean(epoch_costs[-50:]))
            if count % freq_eval == 0:
                dev_score, pred_dev = evaluate_scope(parameters, model.model_path, f_eval, dev_data, id_to_tag, False if pos_tag==0 else True)
                if dev_score > best_dev:
                    best_dev = dev_score
                    print "New best score on dev."
                    print "Saving model to disk..."
                    model.save()
                    test_score, pred_test = evaluate_scope(parameters, model.model_path, f_eval, test_data, id_to_tag, False if pos_tag==0 else True, False)
                    # Store predictions to disk
                    output_predDEV = os.path.join(model.model_path, "best_dev.output")
                    with codecs.open(output_predDEV, 'w', 'utf8') as f:
                        f.write("\n".join(pred_dev))
                    output_predTEST = os.path.join(model.model_path, "best_test.output")
                    with codecs.open(output_predTEST, 'w', 'utf8') as f:
                        f.write("\n".join(pred_test))
                    print "Predictions for the round stored."
        print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))

else:
    print 'Reloading previous model...'
    model.reload()
    test_score, pred_test = evaluate_scope(parameters, model.model_path, f_eval, test_data, id_to_tag, False)

    output_predTEST = os.path.join(model.model_path, "best_test.output")
    with codecs.open(output_predTEST, 'w', 'utf8') as f:
        f.write("\n".join(pred_test))
    print "Test files stored!"
