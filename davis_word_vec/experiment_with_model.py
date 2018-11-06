from gensim.models import word2vec as w2v
import os.path


def test_model(model_location: str, model_name: str):
    print("model: " + model_name)
    model = w2v.Word2Vec.load(model_location)

    nearest_to_good = model.wv.most_similar(positive=['good'], negative=['bad'])
    out_good = ""
    for x in nearest_to_good:
        out_good += x[0] + ","
    print('nearest to \'good\': ' + out_good)

    nearest_to_bad = model.wv.most_similar(positive=['bad'], negative=['good'])
    out_bad = ""
    for x in nearest_to_bad:
        out_bad += x[0] + ","
    print('nearest to \'bad\': ' + out_bad)
    print("\n")





modelSaveLocation = "C:\\Users/davis/PycharmProjects/word_vec/model_storage/model1/w2v_0.1"
model2SaveLocation = "C:\\Users/davis/PycharmProjects/word_vec/model_storage/model2/w2v_0.1"


test_model(modelSaveLocation, "model1")
test_model(model2SaveLocation, "model2")


# model = w2v.Word2Vec.load(modelSaveLocation)
#
# kingVec = model.wv['king']
# manVec = model.wv['man']
# womanVec = model.wv['woman']
#
# outVec = kingVec - manVec + womanVec
# outWord = model.wv.most_similar(positive=[outVec], topn=1)
# outWordCos = model.most_similar_cosmul(positive=[outVec], topn=1)
#
# print("out: " + outWord[0][0] + ", outCosmul: "+outWordCos[0][0])
#
# out2 = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=10)
#
# outStr2 = ""
# for x in out2:
#     outStr2 += x[0] + ","
# print("out2: " + outStr2)
#
#
# nearest_to_good = model.wv.most_similar(positive=['good'], negative=['bad'])
# out3 = ""
# for x in nearest_to_good:
#     out3 += x[0] + ","
# print('nearest to \'good\': ' + out3)
#
#
# nearest_to_bad = model.wv.most_similar(positive=['bad'], negative=['good'])
# out4 = ""
# for x in nearest_to_bad:
#     out4 += x[0] + ","
# print('nearest to \'bad\': ' + out4)


# should_be_wife = model.wv.most_similar(positive=['husband'], negative=['wife'])
# out5 = ""
# for x in should_be_wife:
#     out5 += x[0] + ","
# print('nearest to \'bad\': ' + out5)
