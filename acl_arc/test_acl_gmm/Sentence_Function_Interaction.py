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
		# func_key = self.W_k(func_embed) #shape (N func, word dim)
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

# class SelfAttentionBlock(nn.Module):

# 	def __init__(self, sent_dim ):
# 		super().__init__()
# 		self.sent_dim = sent_dim
# 		self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
# 		self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
# 		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

# 	def forward(self , sent_embed):

# 		sent_query = self.W_q(sent_embed) #shape (N_sent , sent dim)
# 		sent_key = self.W_k(sent_embed) #shape (N func, word dim)
# 		sent_val = self.W_v(sent_embed) #shape (N func , word dim)

# 		z = 1. / math.sqrt(self.sent_dim) * ( torch.matmul( sent_query , torch.transpose(sent_key , 0 , 1 )) ) # shape (N_sent , N_func )
# 		attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (N_sent , N_func )
# 		result = torch.matmul(attention_matrix , sent_val) #shape (N_sent , sent dim )
# 		# print('att ' , attention_matrix[0])
# 		return result #shape (N_sent, sent dim)


# class CrossLeftAttentionBlock(nn.Module):

# 	def __init__(self, sent_dim ):
# 		super().__init__()
# 		self.sent_dim = sent_dim
# 		self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
# 		self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
# 		self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix

# 	def forward(self, toks_emb , func_embed, return_score = False):

# 		tok_query = self.W_q(toks_emb) #shape (N_sent , sent dim)
# 		func_key = self.W_k(func_embed) #shape (N func, word dim)
# 		tok_val = self.W_v(toks_emb) #shape (N tok , word dim)
# 		z1 = ( torch.matmul( tok_query , torch.transpose(func_key , 0 , 1 )) ) # shape (N_tok , N_func )
# 		z = 1. / math.sqrt(self.sent_dim) * z1
# 		attention_matrix = torch.nn.functional.softmax(z, dim = 1) #shape (N_tok , N_func )

# 		result = torch.matmul( torch.transpose(attention_matrix , 0 , 1)  , tok_val ) #N_func, word dim

# 		if return_score == False:
# 			return torch.relu(result) #shape (N_sent, sent dim)
# 		else:
# 			#return score = True, return score of attention weight 
# 			return torch.relu(result),  attention_matrix
	
# class CrossLeftAttentionBlock(nn.Module):

# 	def __init__(self, sent_dim ):
# 		super().__init__()
# 		self.sent_dim = sent_dim
# 		# self.W_q = nn.Linear(self.sent_dim , self.sent_dim) # query matrix
# 		# self.W_k = nn.Linear(self.sent_dim , self.sent_dim) # key matrix
# 		# self.W_v = nn.Linear(self.sent_dim , self.sent_dim) # value matrix
# 		self.W1  = nn.Linear(2* self.sent_dim , self.sent_dim)
# 		self.v = nn.Linear(self.sent_dim , 1)

# 	def forward(self, toks_emb , func_embed, return_score = False):
# 		#toks shape (M ,d)
# 		#func shape (N ,d )
# 		X1 = toks_emb.unsqueeze(1) #(1 , M , d)
# 		Y1 = func_embed.unsqueeze(0) # (N , 1 ,d)
# 		X2 = X1.repeat(1 , func_embed.shape[0],1) # , M n  , d
# 		Y2 = Y1.repeat(toks_emb.shape[0], 1,1) #m , n  , d
# 		Z = torch.cat([X2,Y2],-1) #m , n , 2d
# 		Z = Z.view(-1,Z.shape[-1]) # m x n, 2d
# 		Z = torch.nn.functional.relu(self.W1(Z)) # N xM , d
# 		Z = self.v(Z) #N x M , 1
# 		# Z = Z.squeeze(2) # m x n 
# 		Z = Z.view( toks_emb.shape[0] , func_embed.shape[0] )
# 		att = torch.nn.functional.softmax(Z , dim = 1) # m,n 
# 		result = torch.matmul( torch.transpose(att , 0 , 1)  , toks_emb ) #N_func, word dim

# 		if return_score == False:
# 			return torch.relu(result) #shape (N_sent, sent dim)
# 		else:
# 			#return score = True, return score of attention weight 
# 			return torch.relu(result),  att
