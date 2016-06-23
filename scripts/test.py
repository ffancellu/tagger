#!/usr/bin/env python

import os
import numpy as np
import optparse
import codecs
from collections import OrderedDict
from utils import create_input
import loader
import cPickle

from data_processing import int_processor,ext_processor
from utils import models_path, evaluate_scope
from loader import word_mapping, char_mapping, tag_mapping
from loader import prepare_dataset_scope
from model import Model

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-n", "--test_name", default="default_test",
    help="Name of the test file"
)
optparser.add_option(
    "-l", "--test_lang", default="en",
    help="Language of the test set [en|it|cz]"
)
optparser.add_option(
    "-F", "--folder_name", default="system0",
    help="Name of the folder for the current experiment"
)
opts = optparser.parse_args()[0]

test_sets = opts.test.split(',')
for test_set in test_sets:
    assert os.path.isfile(test_set)

# check whether the embeddings are bilbowa
is_bilbowa = True if opts.folder_name.startswith('bilbowa') else False

folder_path = os.path.abspath(opts.folder_name)
test_lang = opts.test_lang

# Initialize model
model = Model(model_path=folder_path, bilbowa=is_bilbowa)
print "Model location: %s" % model.model_path

# Load parameters, no load mappings
pos_tag = model.parameters['pos_tag']

# get voc and voc_dic from pickle anyway
with open(os.path.join(folder_path,'train_dev.pkl'),'rb') as data_pkl:
    _, _, voc, voc_inv = cPickle.load(data_pkl)

if test_lang != "en" and is_bilbowa:
    # take voc from pickle and create
    w2idxs, idxs2w = int_processor.get_test_dicts(test_sets, True, False, test_lang, model.parameters['tag_scheme'])
    voc['w2idxs'] = w2idxs
    voc_inv['idxs2w'] = idxs2w

# Load dictionaries from pickle
test_lex, test_tags, test_tags_uni, test_cue, _, test_y = int_processor.load_test(test_sets, voc, True, False, test_lang, model.parameters['tag_scheme'])

# NOT RELEVANT FOR THE MOMENT
dico_chars, char_to_id, id_to_char = char_mapping([[voc_inv['idxs2w'][t] for t in idx_sent] for idx_sent in test_lex])

test_data = prepare_dataset_scope(
    [[voc_inv['idxs2w'][t] for t in idx_sent] for idx_sent in test_lex],
    test_lex,
    test_cue,
    test_tags_uni if pos_tag == 2 else test_tags,
    test_y,
    char_to_id)

print "%i sentences in test." % len(test_data)

word_to_id = voc['w2idxs']

id_to_word = voc_inv['idxs2w']
id_to_tags = voc_inv['idxs2tuni'] if pos_tag == 2 else voc_inv['idxs2t']
id_to_y = voc_inv['idxs2y']

# Save the mappings to disk
print 'Set new mappings...'
model.set_mappings(id_to_word, id_to_char, id_to_tags, id_to_y)


print "Model built!"


# *******INITIALIZE THE MODEL********
# in the case of Bilbowa we need to initialize a matrix n x emb_dim

f_train, f_eval = model.build(training=False, **model.parameters)


# print 'Reloading previous model...'
# model.reload()
# test_score, pred_test = evaluate_scope(parameters, model.model_path, f_eval, test_data, id_to_tag, False)

# output_predTEST = os.path.join(model.model_path, "best_test.output")
# with codecs.open(output_predTEST, 'w', 'utf8') as f:
#     f.write("\n".join(pred_test))
# print "Test files stored!"