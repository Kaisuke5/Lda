#coding:utf-8
import numpy as np
from numpy import ones,zeros
import Vocabulary
import MeCab

import six


class lda():

	def __init__(self,alpha,beta,iteraion,K,docs,V,voc):
		self.alpha=alpha
		self.beta=beta
		self.K=K
		self.docs=docs
		self.M=len(docs)
		self.V=V
		self.voc=voc
		self.iteraion=iteraion

	
		
		
		self.size_d=len(self.docs)
		self.topic=[]
		self.n_d_z=np.zeros((self.M,self.K))+self.alpha	#of tokens with topic z in d
		self.n_z_w=np.zeros((self.K,self.V))+self.beta	#count of words in topic z and w
		self.n_z=np.zeros(self.K)+self.V*beta		#count of words in topic z


		for m,doc in enumerate(self.docs):
			#doc=["kai","shun",..]
			z_m=np.random.randint(0,self.K,len(doc))
			self.topic.append(z_m)
			
			for z,word in zip(z_m,doc):
				self.n_d_z[m,z]+=1
				self.n_z_w[z,word]+=1
				self.n_z[z]+=1

	def reinit(self):
		self.voc.mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")


	def infer(self):
		for m,doc in enumerate(self.docs):
			z_n=self.topic[m] #in dm: [cat1 cat3 ...] (N)
			n_m_z=self.n_d_z[m] #[3 2 5...] (K)

			
			for n,w in enumerate(doc):
				
				z=z_n[n]
				n_m_z[z]-=1
				self.n_z_w[z,w]-=1
				self.n_z[z]-=1

				
				
				#p_z=n_m_z*self.n_z_w[:,w]/(self.n_z*sum(self.n_d_z[m]))
				p_z = self.n_z_w[:, w] * n_m_z / self.n_z
				
				new_z=np.random.multinomial(1,p_z/p_z.sum()).argmax()
				
				z_n[n]=new_z
				n_m_z[new_z]+=1
				self.n_z_w[new_z,w]+=1
				self.n_z[new_z]+=1

		return


	#train

	def train(self):
		tmp=0
		for i in range(self.iteraion):
			print "iter:",i
			self.infer()
			L=self.perplexity()
			print "L(i-1)-L(i):",tmp-L," perplexity:",L
			tmp=L


	def train2(self):
		tmp=0
		num_save = 10
		pickle_name = "./data/model/wiki_lda_model_"
		output_name = './data/output/word_cluster_num_'

		for i in range(self.iteraion):

			self.infer()
			L=self.perplexity()

			if i % num_save == 0:
				print "L(i-1)-L(i):",tmp-L," perplexity:",L

				pname = pickle_name + str(i) + ".dump"
				oname = output_name + str(i) + ".txt"


				with open(pname, 'wb') as f:
					six.moves.cPickle.dump(l, f, -1)
					print 'Done %s' % pname

				self.word_clustering(filename=oname)

			tmp=L






	# to write ouput class to file
	def word_clustering(self,num=30,filename='output.txt'):
		f=open(filename,"w")
		data = []

		for k in range(self.K):
			pw_k=self.n_z_w[k]/self.n_z[k]
			index=np.argsort(pw_k)
			
			f.write("cluster:"+str(k)+"\n")

			for i,idx in enumerate(index[-num:]):

				line=str(self.voc.words[idx])+" "+str(self.n_z_w[k][idx]/self.n_z[k])+"\n"
				data.append(line)
				f.write(line)
			f.write("\n")
		f.close()
		return data


	def doc_clustering(self,docs,fileoption=True):
		f=open("output2","w")
		for i,doc in enumerate(docs):
			p=[0 for j in range(self.K)]
			for k in range(self.K):
				for d in doc:
					p[k]+=np.log(self.n_z_w[k][d])

			max_class=p.index(max(p))
			if fileoption: f.write("document "+str(i)+":"+str(max_class)+"\n")
			else: print

		f.close()


	def clustering(self,sentence):
		doc = []
		max_size = len(self.voc.words)
		doc = self.voc.doc_to_ids(sentence,add_option=False)
		doc = filter(lambda x:x<max_size,doc)



		p=[0 for j in range(self.K)]
		for k in range(self.K):
			for d in doc:
				p[k]+=np.log(self.n_z_w[k][d])

		max_class=p.index(max(p))
		return "class:",max_class

	def wordlist(self):
		return self.n_z_w/self.n_z[:,np.newaxis]


	def perplexity(self,docs=None):
		if docs == None : docs=self.docs
		phi=self.wordlist()
		log_per=0
		N=0
		Kalpha=self.K*self.alpha
		for m,doc in enumerate(docs):
			theta=self.n_d_z[m]/ (len(self.docs[m]) + Kalpha)
			for w in doc:
				log_per-=np.log(np.inner(phi[:,w],theta))
			N+=len(doc)
		return np.exp(log_per/N)


if __name__=="__main__":
	import optparse
	p=optparse.OptionParser()
	p.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
	p.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
	p.add_option("-k", dest="k", type="int", help="topic K", )
	p.add_option("-i", dest="iteraion", type="int", help="iteraion" ,default=200)
	p.add_option("-f", dest="filename", help="corpus filename")
	p.add_option("-l", dest="language", help="language",default="en")

	(op, args) = p.parse_args()
	if not (op.filename and op.language and op.k):p.error("valid options\nyou must use options -k -f -l\nif you check options,you should type input -h")


	v=Vocabulary.Vocabulary(language=op.language)
	dd=v.make_corpus(op.filename)
	
	l=lda(op.alpha,op.beta,op.iteraion,op.k,dd,len(v.words),v)
	l.train2()
	l.word_clustering()
	l.doc_clustering(dd)
	print l.clustering("かいくん麻薬")
	print l.clustering("かいくん安保反対")
	fname = "lda_model.dump"
	with open(fname, 'wb') as output:
		six.moves.cPickle.dump(l, output, -1)
		print 'Done %s' % fname




	
	



