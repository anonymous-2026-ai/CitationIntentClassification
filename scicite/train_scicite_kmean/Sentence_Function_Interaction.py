from transformers import BertModel
import torch.nn as nn
import torch 
import math 

class Sentence_Function_Interaction(nn.Module):

	def __init__(self,  word_emb_dim, sent_emb_dim, n_iter , device):
		
		super().__init__()
		self.device = device
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.sent_emb_dim = sent_emb_dim
		self.n_iter = n_iter
		self.linear_pool = nn.Linear(self.word_emb_dim , self.sent_emb_dim , bias=True)
		self.sent2func = CrossAttentionBlock(self.word_emb_dim)
		self.func2sent = CrossAttentionBlock(self.word_emb_dim)
		# self.sent_fun_atten = SelfAttentionBlock(self.sent_emb_dim)
		self.ln = nn.LayerNorm(self.sent_emb_dim)
		# self.tok2func = CrossLeftAttentionBlock(self.word_emb_dim)

	def forward(self, sents_emb , func_embed):
		#sents_emb (N_sent , word dim) 
		#func emb (N_func, word dim)
		sents_hid  = sents_emb 
		func_hid = func_embed
		# for i in range(self.n_iter):
		# 	#use N iter for getting the hidden state of sentence 
		# 	sents_hid  = sents_hid + self.func2sent(func_hid , sents_hid) # N sent , word dim 
		# 	func_hid = func_hid + self.sent2func(sents_hid , func_hid) # N func , word dim
		# return sents_hid , func_hid 
		for i in range(self.n_iter):
			# if i != self.n_iter - 1 :
			# 	sents_hid  = sents_hid + self.func2sent(func_hid , sents_hid) # N sent , word dim 
			# 	func_hid = func_hid + self.sent2func(sents_hid , func_hid) # N func , word dim
			# else:
			out_fun2sent_hid, out_fun2sent_att = self.func2sent(func_hid , sents_hid, return_score = True )
			sents_hid  = sents_hid + out_fun2sent_hid
			func_hid = func_hid + self.sent2func(sents_hid , func_hid  )

		return sents_hid , func_hid , out_fun2sent_att

	# def forward(self, sents_emb , func_embed):
	# 	#sents_emb (N_sent , word dim) 
	# 	#func emb (N_func, word dim)
	# 	sents_hid  = sents_emb 
	# 	func_hid = func_embed

	# 	sent_fun = torch.cat( [sents_emb , func_embed] , dim = 0  ) #num sent + num func ,  word dim
	# 	# for i in range(self.n_iter):
	# 	# 	#use N iter for getting the hidden state of sentence 
	# 	# 	sents_hid  = sents_hid + self.func2sent(func_hid , sents_hid) # N sent , word dim 
	# 	# 	func_hid = func_hid + self.sent2func(sents_hid , func_hid) # N func , word dim
	# 	# return sents_hid , func_hid 
	# 	for i in range(self.n_iter):
	# 		sent_fun  = self.ln(sent_fun + self.sent_fun_atten(sent_fun))
			
	# 	sents_hid = sent_fun[: sents_hid.shape[0], : ]
	# 	func_hid  = sent_fun[sents_hid.shape[0] : , :]
		
	# 	return sents_hid , func_hid
	
	def tokens2func(self, toks_embed , func_embed , return_score= True):
		func_hid = func_embed
		# toks_hid = toks_embed

		for i in range(self.n_iter):
			add_hid , att  = self.tok2func( toks_embed , func_hid, return_score)
			func_hid = self.ln(func_hid + add_hid)

		return func_hid , att
	
class CrossAttentionBlock(nn.Module):

	def __init__(self, sent_dim ):
		super().__init__()
		self.sent_dim = sent_dim
		self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
		self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

	def forward(self, func_embed , sent_embed, return_score = False):

		sent_query = sent_embed #shape (N_sent , sent dim)
		func_key = func_embed #shape (N func, word dim)
		func_val = self.W_v(func_embed) #shape (N func , word dim)

		z1 = ( torch.matmul( sent_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_sent , N_func )
		z = 1. / math.sqrt(self.sent_dim) * z1
		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_func )
		result = torch.matmul(attention_matrix , func_val) #shape (N_sent , sent dim )

		if return_score == False:
			return result #shape (N_sent, sent dim)
		else:
			#return score = True, return score of attention weight 
			return result,  z.tolist()
