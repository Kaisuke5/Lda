import pickle
from Lda import lda
import Vocabulary
import MeCab




v = Vocabulary.Vocabulary(language="ja")
dd=v.make_corpus("anpo.txt")
l = lda(0.5,0.5,100,2,dd,len(v.words),v)

l.train2()

