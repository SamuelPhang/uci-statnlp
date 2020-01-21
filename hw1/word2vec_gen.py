from gensim.models import Word2Vec
from gensim.utils import any2utf8
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

def read_files(tarfname):
    """Read all files from the tar file as a list of documents"""
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")

    class Data:
        pass

    unlabeled = Data()
    unlabeled.data = []
    #unlabeled.fnames = []
    for m in tar.getmembers():
        if ".txt" in m.name:
            #unlabeled.fnames.append(m.name)
            unlabeled.data.append(read_instance(tar, m.name))
    tar.close()
    return unlabeled.data

def read_instance(tar, ifname):
    inst = tar.getmember(ifname)
    ifile = tar.extractfile(inst)
    content = ifile.read().strip()
    return content

if __name__ == "__main__":
    print("Reading files")
    tarfname = "data/speech.tar.gz"
    docs = read_files(tarfname)
    lmtzr = WordNetLemmatizer()
    print("Lemmatizing and Tokenizing")
    lemmatized = [[lmtzr.lemmatize(word) for word in word_tokenize(d.decode("utf-8"))] for d in docs]
    print("Computing Word2Vec Matrix")
    wv = Word2Vec(lemmatized)
    wv.save("word2vec.model")