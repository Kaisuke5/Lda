#coding:utf-8
import re
import MeCab 
import json
from tqdm import tqdm
import six

def pp(obj):
  if isinstance(obj, list) or isinstance(obj, dict):
    orig = json.dumps(obj, indent=4)
    print eval("u'''%s'''" % orig).encode('utf-8')
  else:
    print obj


def load_corpus(filename):
	
	corpus=[]

	for line in open(filename,"r"):	
		corpus.append(line.split())
	return corpus


stopwords=[]
stopwords.append("a,s,able,about,above,according,accordingly,across,actually,after,afterwards,again,against,ain,t,all,allow,allows,almost,alone,along,already,also,although,always,am,among,amongst,an,and,another,any,anybody,anyhow,anyone,anything,anyway,anyways,anywhere,apart,appear,appreciate,appropriate,are,aren,t,around,as,aside,ask,asking,associated,at,available,away,awfully,be,became,because,become,becomes,becoming,been,before,beforehand,behind,being,believe,below,beside,besides,best,better,between,beyond,both,brief,but,by,c,mon,c,s,came,can,can,t,cannot,cant,cause,causes,certain,certainly,changes,clearly,co,com,come,comes,concerning,consequently,consider,considering,contain,containing,contains,corresponding,could,couldn,t,course,currently,definitely,described,despite,did,didn,t,different,do,does,doesn,t,doing,don,t,done,down,downwards,during,each,edu,eg,eight,either,else,elsewhere,enough,entirely,especially,et,etc,even,ever,every,everybody,everyone,everything,everywhere,ex,exactly,example,except,far,few,fifth,first,five,followed,following,follows,for,former,formerly,forth,four,from,further,furthermore,get,gets,getting,given,gives,go,goes,going,gone,got,gotten,greetings,had,hadn,t,happens,hardly,has,hasn,t,have,haven,t,having,he,he,s,hello,help,hence,her,here,here,s,hereafter,hereby,herein,hereupon,hers,herself,hi,him,himself,his,hither,hopefully,how,howbeit,however,i,d,i,ll,i,m,i,ve,ie,if,ignored,immediate,in,inasmuch,inc,indeed,indicate,indicated,indicates,inner,insofar,instead,into,inward,is,isn,t,it,it,d,it,ll,it,s,its,itself,just,keep,keeps,kept,know,knows,known,last,lately,later,latter,latterly,least,less,lest,let,let,s,like,liked,likely,little,look,looking,looks,ltd,mainly,many,may,maybe,me,mean,meanwhile,merely,might,more,moreover,most,mostly,much,must,my,myself,name,namely,nd,near,nearly,necessary,need,needs,neither,never,nevertheless,new,next,nine,no,nobody,non,none,noone,nor,normally,not,nothing,novel,now,nowhere,obviously,of,off,often,oh,ok,okay,old,on,once,one,ones,only,onto,or,other,others,otherwise,ought,our,ours,ourselves,out,outside,over,overall,own,particular,particularly,per,perhaps,placed,please,plus,possible,presumably,probably,provides,que,quite,qv,rather,rd,re,really,reasonably,regarding,regardless,regards,relatively,respectively,right,said,same,saw,say,saying,says,second,secondly,see,seeing,seem,seemed,seeming,seems,seen,self,selves,sensible,sent,serious,seriously,seven,several,shall,she,should,shouldn,t,since,six,so,some,somebody,somehow,someone,something,sometime,sometimes,somewhat,somewhere,soon,sorry,specified,specify,specifying,still,sub,such,sup,sure,t,s,take,taken,tell,tends,th,than,thank,thanks,thanx,that,that,s,thats,the,their,theirs,them,themselves,then,thence,there,there,s,thereafter,thereby,therefore,therein,theres,thereupon,these,they,they,d,they,ll,they,re,they,ve,think,third,this,thorough,thoroughly,those,though,three,through,throughout,thru,thus,to,together,too,took,toward,towards,tried,tries,truly,try,trying,twice,two,un,under,unfortunately,unless,unlikely,until,unto,up,upon,us,use,used,useful,uses,using,usually,value,various,very,via,viz,vs,want,wants,was,wasn,t,way,we,we,d,we,ll,we,re,we,ve,welcome,well,went,were,weren,t,what,what,s,whatever,when,whence,whenever,where,where,s,whereafter,whereas,whereby,wherein,whereupon,wherever,whether,which,while,whither,who,who,s,whoever,whole,whom,whose,why,will,willing,wish,with,within,without,won,t,wonder,would,would,wouldn,t,yes,yet,you,you,d,you,ll,you,re,you,ve,your,yours,yourself,yourselves,zero".split(','))
stopwords.append("僕,私".split(","))



def make_japanese_text(input_filename,output_filename,pos_lst = ["名詞"]):
	mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
	s = ''

	for line in open(input_filename,"r"):

		node = mecab.parseToNode(line)
		while node:
			feature = node.feature.split(",")[0]

			if feature in pos_lst:
				s += node.surface + " "

			node=node.next
		s += '\n'

	with open(output_filename,"w") as f:
		f.write(s)

	return










class Vocabulary():

	
	def __init__(self,language="en"):
		self.words=[]
		self.word_id={}
		self.docfreq=[]
		self.language=0
		#self.mecab=MeCab.Tagger("mecabrc")

		#if you have a ipadic
		self.mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")

		if language=="ja": self.language=1
		
	def __getstate__(self):
		odict = self.__dict__.copy()
		del odict['mecab'] # pickle 時に保存しないよう、消してしまう
		return odict

	def __setstate__(self, odict):
		self.__dict__.update(odict)
		self.mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")




	def word_to_id(self,word,add_option=True):
		if word in stopwords[self.language]: return None
		#if self.language==0 and not re.match(r'[a-z]+$', word): return None
		if re.match(r'[1-9]',word): return None



		if word not in self.words:
			if not add_option: return None
			self.word_id[word]=len(self.words)
			self.words.append(word)
			self.docfreq.append(1)
		else:
			self.docfreq[self.word_id[word]]+=1
		return self.word_id[word]


	


	def doc_to_ids(self,doc,add_option=True):
		list=[]
		
		if self.language==1: doc=self.mecabdoc(doc)
		else: doc=doc.split()
		
		for word in doc:
			
			id=self.word_to_id(word,add_option=add_option)

			if id !=None:
				list.append(id)
		return list

	def getsentence(self,docs):
		line=[]
		for doc in docs:
			line.append([self.__getitem__(id) for id in doc])
		pp(line)


	def __getitem__(self,v):
		return self.words[v]



	def make_corpus(self,filename):
		docs=[]

		for line in tqdm(open(filename,"r")):

			doc=self.doc_to_ids(line)
			if len(doc)> 10: docs.append(doc)
		return docs
			

	def mecabdoc(self,doc,option=["名詞","形容詞"]):
		
		node=self.mecab.parseToNode(doc)

		fixeddoc=[]
		while node:
			feature=node.feature.split(",")[0]
			if feature in option:
				fixeddoc.append(node.surface)

			node=node.next

		return fixeddoc







if __name__=="__main__":
	WIKI_DATA_PATH = '/home/ec2-user/lda_project/text/result.txt'
	WIKI_DATA_PATH = 'data/test_corpus2.txt'
	# VOC_PICKLE = 'wiki_vocaburary.dump'
	#
	#
	#WIKI_DATA_PATH = "anpo.txt"
	v=Vocabulary(language="en")
	docs=v.make_corpus(WIKI_DATA_PATH)
	print pp(docs)
	#
	# with open(VOC_PICKLE, 'wb') as output:
	# 	six.moves.cPickle.dump([docs, v], output, -1)
	# 	print 'Done %s' % VOC_PICKLE
	#
	#
	# print docs[0]
	#make_japanese_text("~/lda_project/result.txt",'wiki_japanese_corpus.txt')
