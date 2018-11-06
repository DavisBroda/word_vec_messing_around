from gensim.models import word2vec as w2v
import os.path
import unicodedata as uCode
import gensim
import preprocess as pre
import model_utils as m_util
from itertools import chain

# number of times to run through training data
iterations = 20
# minimum number of times a word needs to show up to be in vocabulary
min_occurances = 5
# size of the word vector generated for a word
word_vec_size = 50
# workers to use while training
paralellism_factor = 4

rootDir = "C:\\Users/davis/PycharmProjects/word_vec/data"
abcRuralData = rootDir + "/ABC_News_Archive/rural.txt"
abcScienceData = rootDir + "/ABC_News_Archive/science.txt"
abcCombinedData = rootDir + "/ABC_News_Archive/combined_abc_fix.txt"

gTweetData = rootDir + "/Graeham_tweet_data/toxic-comments/train.csv"
davis_comment_only = rootDir + "/Graeham_tweet_data/davis_preprocessed/comment_text_only.txt"

amazon_comment_location = rootDir + "/Comparative_Sentence_Dataset/labeledSentences.txt"
europarliment_location = rootDir + "/europarl_raw/english/*"
gutenburg_location = rootDir + "/gutenberg/*.txt"
state_union_location = rootDir + "/state_union/*.txt"


modelSaveLocation = "C:\\Users/davis/PycharmProjects/word_vec/model_storage/model1/w2v_0.1"

# replace with less specific reader later on
tweets = w2v.Text8Corpus(gTweetData)

abc_lines = pre.read_input_file(abcCombinedData)

# non UTF-8 char apparently causing gailure in below
model = gensim.models.Word2Vec(
    abc_lines, iter=iterations, min_count=min_occurances, size=word_vec_size, workers=paralellism_factor)

m_util.additional_training(model, amazon_comment_location)
m_util.additional_training(model, europarliment_location)
m_util.additional_training(model, gutenburg_location)
m_util.additional_training(model, state_union_location)

m_util.save_model(model, modelSaveLocation)

#seeing if different result if combining before creating model
model2Location = "C:\\Users/davis/PycharmProjects/word_vec/model_storage/model2/w2v_0.1"

model2Lines = chain(
    pre.read_input_file(abcCombinedData),
    pre.read_input_file(amazon_comment_location),
    pre.read_input_file(europarliment_location),
    pre.read_input_file(gutenburg_location),
    pre.read_input_file(state_union_location)
)

model2LineCount = chain(
    pre.get_file_line_count(abcCombinedData),
    pre.get_file_line_count(amazon_comment_location),
    pre.get_file_line_count(europarliment_location),
    pre.get_file_line_count(gutenburg_location),
    pre.get_file_line_count(state_union_location)
)

model2 = gensim.models.Word2Vec(
    model2Lines, iter=iterations, min_count=min_occurances, size=word_vec_size, workers=paralellism_factor)

m_util.save_model(model2, model2Location)

