from gensim.models import word2vec as w2v
import preprocess as pre
import os.path


def additional_training(model: w2v.Word2Vec, file_location: str):
    lines = pre.read_input_file(file_location)
    line_count = pre.get_file_line_count(file_location)
    model.train(lines, total_examples=line_count, epochs=model.epochs)
    return model


def save_model(model: w2v.Word2Vec, location: str):
    if os.path.exists(location):
        os.remove(location)

    model.save(location)
