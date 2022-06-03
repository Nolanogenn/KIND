import pandas as pd
from conllu import parse
import sys
import spacy
from spacy.attrs import NORM, ORTH, LEMMA, POS
from spacy.symbols import prep, det
from spacy.matcher import Matcher
from spacy.language import Language
import re

nlp = spacy.load('it_core_news_sm')
df = pd.read_csv(sys.argv[1], sep='\t', header=None)
#we need to incorporate some exceptions for the tokenizer
special_cases = [
        (u'dell’', [{ORTH: u'dell’', NORM: u'dello'}]),
        (u'dell’', [{ORTH: u'dell’', NORM: u'della'}]),
        (u'dall’', [{ORTH: u'dall’', NORM: u'dalla'}]),
        (u'sull’', [{ORTH: u'sull’', NORM: u'sulla'}]),
        (u'sull’', [{ORTH: u'sull’', NORM: u'sullo'}]),
        (u'sull’', [{ORTH: u'sull’', NORM: u'sulle'}]),
        (u'sull’', [{ORTH: u'sull’', NORM: u'sugli'}]),
        (u'dall’', [{ORTH: u'dall’', NORM: u'dallo'}]),
        (u'c’', [{ORTH: u'c’', NORM: u'ci'}]),
        (u'd’', [{ORTH: u'd’', NORM: u'di'}]),
        (u'l’', [{ORTH: u'l’', NORM: u'lo'}]),
        (u'l’', [{ORTH: u'l’', NORM: u'la'}]),
        (u'l’', [{ORTH: u'l’', NORM: u'le'}]),
        (u'l’', [{ORTH: u'l’', NORM: u'il'}]),
        (u'all’', [{ORTH: u'all’', NORM: u'allo'}]),
        (u'all’', [{ORTH: u'all’', NORM: u'alla'}]),
        (u'un’', [{ORTH: u'un’', NORM: u'una'}]),
        (u'nell’', [{ORTH: u'nell’', NORM: u'nella'}]),
        (u'nell’', [{ORTH: u'nell’', NORM: u'nello'}]),
        (u'j’', [{ORTH: u'j’', NORM: u'je'}]),
        (u'de’', [{ORTH: 'de’', NORM: u'del'}]),
        ######
        (u'j\'', [{ORTH: u'j\'', NORM: u'je'}]),
        (u'de\'', [{ORTH: u'de\'', NORM: u'del'}]),
        (u'dell\'', [{ORTH: u'dell\'', NORM: u'dello'}]),
        (u'Rainey\'', [{ORTH: u'Rainey\'', NORM: u'Rainey\''}]),
        (u'dell\'', [{ORTH: u'dell\'', NORM: u'della'}]),
        (u'dall\'', [{ORTH: u'dall\'', NORM: u'dalla'}]),
        (u'sull\'', [{ORTH: u'sull\'', NORM: u'sulla'}]),
        (u'sull\'', [{ORTH: u'sull\'', NORM: u'sullo'}]),
        (u'sull\'', [{ORTH: u'sull\'', NORM: u'sulle'}]),
        (u'sull\'', [{ORTH: u'sull\'', NORM: u'sugli'}]),
        (u'dall\'', [{ORTH: u'dall\'', NORM: u'dallo'}]),
        (u'c\'', [{ORTH: u'c\'', NORM: u'ci'}]),
        (u'd\'', [{ORTH: u'd\'', NORM: u'di'}]),
        (u'l\'', [{ORTH: u'l\'', NORM: u'lo'}]),
        (u'l\'', [{ORTH: u'l\'', NORM: u'la'}]),
        (u'l\'', [{ORTH: u'l\'', NORM: u'le'}]),
        (u'l\'', [{ORTH: u'l\'', NORM: u'il'}]),
        (u'all\'', [{ORTH: u'all\'', NORM: u'allo'}]),
        (u'all\'', [{ORTH: u'all\'', NORM: u'alla'}]),
        (u'un\'', [{ORTH: u'un\'', NORM: u'un'}]),
        (u'nell\'', [{ORTH: u'nell\'', NORM: u'nella'}]),
        (u'nell\'', [{ORTH: u'nell\'', NORM: u'nello'}]),
        #######
        (u'egr.', [{ORTH: u'egr.', NORM: u'egregio'}]),
        (u'sig.', [{ORTH: u'sig.', NORM: u'signor'}]),
        ]

to_find = ["'", "'", ".", "-", "%", "#", "’", "°"]
for i, r in df.iterrows():
    if any(j in r[0] for j in to_find) and len(r[0]) >= 2:
        word = u'{}'.format(r[0].replace("'", "\'"))
        rule = (word, [
            {
                ORTH : word,
                NORM: word
                }
            ])
        special_cases.append(rule)


for case_ in special_cases:
    nlp.tokenizer.add_special_case(case_[0], case_[1])
    first_letter_uppercase = u"{}{}".format(case_[0][0].upper(),case_[0][1:])
    for_uppercase = [
                first_letter_uppercase,
                [
                    {
                        ORTH: first_letter_uppercase,
                        NORM: case_[1][0][67]
                    }
            ]]
    nlp.tokenizer.add_special_case(for_uppercase[0], for_uppercase[1])

###################
file = sys.argv[1]
file = open(file).readlines()
fixed_file_with_id = ''
for enum, line in enumerate(file):
    if line != '\n':
        new_line = f"{enum}\t{line}"
        fixed_file_with_id += new_line
    else:
        fixed_file_with_id += line

sentences = parse(fixed_file_with_id)

sentences_dict = {}
columns = [
        'id', 
        'form',
        'lemma',
        'upos', 
        'xpos',
        'feats',
        'head',
        'deprel',
        'deps',
        'label',
        'misc'
        ]

file_conll_format = ''

for enum, sentence in enumerate(sentences):

    tokens_id = []
    tokens_form = []
    tokens_upos = []
    tokens_xpos = []
    tokens_feats = []
    tokens_head = []
    tokens_deprel = []
    tokens_deps = []
    tokens_label = []
    tokens_misc = []
    tokens_lemma = []

    sentence_string = ''
    sentence_tokens = []

    spacy_ids_to_conll_id = {}
    for enum, token in enumerate(sentence[:20]):
        token_id = token['id']
        token_form = token['form']
        token_label = token['lemma']
        
        spacy_ids_to_conll_id[enum] = token_id
        tokens_id.append(token_id)
        tokens_form.append(token_form)
        tokens_label.append(token_label)

        sentence_string += token_form
        sentence_tokens.append(token_form)
        sentence_string += ' '
    
    sentence_string = sentence_string[:-1]
    doc = nlp(sentence_string)

    for enum, spacy_token in enumerate(doc):
        token_lemma = spacy_token.lemma_
        token_upos = spacy_token.pos_
        #TODO: implement xpos somehow
        token_xpos = '_'
        token_feats = spacy_token.morph if len(spacy_token.morph) > 0 else '_'
        #TODO: head should point to an id from the spacy_ids_to_conll_ids created above
        token_head = spacy_token.head
        token_deprel = spacy_token.dep_
        #TODO: implement enhanced dep
        token_deps = '_'
        token_misc = '_'

        tokens_lemma.append(token_lemma)
        tokens_upos.append(token_upos)
        tokens_xpos.append(token_xpos)
        tokens_feats.append(token_feats)
        tokens_head.append(token_head)
        tokens_deprel.append(token_deprel)
        tokens_deps.append(token_deps)
        tokens_misc.append(token_misc)
    
    for i, token_id in enumerate(tokens_id):
        file_conll_format += f"{token_id}\t{tokens_form[i]}\t{tokens_lemma[i]}\t{tokens_upos[i]}\t{tokens_xpos[i]}\t{tokens_feats[i]}\t{tokens_head[i]}\t{tokens_deprel[i]}\t{tokens_deps[i]}\t{tokens_misc[i]}\t{tokens_label[i]}\n"
    file_conll_format += "\n"

print(file_conll_format)
