import vocabulary
import lda

filename="train.txt"
corpus = vocabulary.load_corpus(filename)
voca = vocabulary.Vocabulary(corpus)
docs = [voca.doc_to_ids(doc) for doc in corpus]
K=5
alpha=1
beta=1
l = lda.LDA(5, 0.5, 0.5, docs, len(voca.words))
print ("corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.words), K, alpha, beta))

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
lda.lda_learning(l, 200, voca)
