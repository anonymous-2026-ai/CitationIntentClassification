from transformers import BertModel
import torch.nn as nn
import torch 
from Sentence_Function_Interaction import * 
from Word_Function_Interaction import  *
from Doc_representation import * 
from kmeans_pytorch import kmeans
import numpy as np
# from sklearn.cluster import KMeans

class Model(nn.Module):

	def __init__(self, pretrained_model, word_emb_dim, sent_emb_dim, n_iter_sent , n_iter_word, device):
		super().__init__()
		self.device = device
		self.pretrained_model = pretrained_model
		self.word_emb_dim = word_emb_dim # for roberta large, it is set by 1024
		self.sent_emb_dim = sent_emb_dim # this value usually is set by 64 
		self.linear_pool = nn.Linear(self.word_emb_dim , self.sent_emb_dim , bias=True)
		self.sent_func_interaction = Sentence_Function_Interaction(word_emb_dim, sent_emb_dim, n_iter = n_iter_sent, device = self.device)
		self.doc_pool = Doc_representation(self.word_emb_dim, n_func = 6 , device = self.device)
		self.word_func_interaction = Word_Function_Interaction(word_emb_dim , n_iter = n_iter_word, device=self.device )
		# self.W_q_sent = nn.Linear(self.word_emb_dim , 1) # value matrix
		# self.W_v_sent = nn.Linear(self.word_emb_dim , self.word_emb_dim) # value matrix

	def pool_sent_embed(self , last_hidden_state , features,  batch_index):
		batch_feature = [features[t] for t in batch_index]
		ques_embed , list_sent_emb , batch_bound_sents = self.pool_embed(last_hidden_state , batch_feature) #init embed contain question embedding and list of sentence embedding 
		return ques_embed , list_sent_emb , batch_bound_sents

	def forward(self, chunks_idx , tok2sent_ind ,  labels_idx , tok2label, tokenizer , print_att = False):
		# print('---')
		#chunk idx: list of tokens'index in the citation context
		#--> need to add the cls token and sep token into start/end position of original sequence 
		#func embed shape (N_function , func hidden size)
		# print('tok2label ', tok2label)
		# print('tok2sentidx ', tok2sent_ind)
		input_tokens =  [  [102] + t +  [103]    for t in chunks_idx] #0 vs 2 for RoBerTA	
		#len list = number of chunk 

		token_hidden_state = self.forward_pretrained_tok(input_tokens) 

		toks_embed = self.extract_tok_label_emb(token_hidden_state)

		if toks_embed.shape[0] >= 2 :

			cluster_ids_x, cluster_centers = kmeans(
			X=toks_embed, num_clusters=2, distance='euclidean', device=torch.device('cuda:0'), tol = 1e-3, tqdm_flag = False
			)		
		else:
			cluster_centers = torch.rand(2 , toks_embed.shape[1])

		# if toks_embed.shape[0] >= 7 :
		# 	X = toks_embed.detach().cpu().numpy()
		# 	kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(X)
		# 	cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
		# else:
		# 	cluster_centers = torch.rand(7 , toks_embed.shape[1])

		cluster_centers = cluster_centers.to(self.device)

		# # tok_hid , func_hid , tok_score= self.sent_func_interaction(toks_embed , func_embed )]
		# # func_hid , att= self.sent_func_interaction.tokens2func(toks_embed , func_embed, return_score=True)
		if toks_embed.shape[0] != len(tok2sent_ind):
			print('error .......')
			return 
		
		func_emb = cluster_centers		
		toks_hid , func_hid = self.word_func_interaction(toks_embed , cluster_centers)

		sent_embed = self.sent_pool(toks_hid , tok2sent_ind)

		sents_hid , _ , _ = self.sent_func_interaction(sent_embed , func_hid )

		doc_emb= self.doc_pool(sents_hid) #(word dim)

		doc_logit = self.doc_pool.compute_doc_logit(doc_emb) # (N_func) ---> use for multi binary loss 

		return doc_logit.squeeze(0) 

	def get_label_hidden(self, labels_idx):
		label_idx = labels_idx[0]
		result = [] 
		for idx in label_idx:
			result.append(torch.tensor(self.pretrained_model.embeddings.word_embeddings(torch.tensor(idx).to(self.device))))
		return torch.stack(result, dim = 0)

	def extract_tok_label_emb(self,tok_hidden_state):
		#tok hidden state is list of tensor , each tensor shape (num label , num token , hidden dim  )
		list_tok_emb = [] 
		for t in tok_hidden_state: 
			#t shape (num label , num token+  1 +2  , hidden dim )
			tok_emb =  t[: , 1:-1,:] #shape ( num label, num tok in context , word dim )
			func_emb = t[ : , -2, :].squeeze(1) # shape (num label, word dim)
			
			tok_emb = torch.mean(tok_emb , dim = 0 ) # shape (num tok in context , word dim)
			
			list_tok_emb.append(tok_emb)

		all_tok_emb = torch.cat(list_tok_emb , dim = 0) #shape (num tok in doc, word dim )
		return all_tok_emb 

	def get_label_hidden(self, labels_idx):
		label_idx = labels_idx[0]
		result = [] 
		for idx in label_idx:
			result.append(torch.tensor(self.pretrained_model.embeddings.word_embeddings(torch.tensor(idx).to(self.device))))
		return torch.stack(result, dim = 0)	

	def forward_attention(self, input ):
		#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
		# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
		output = []
		for i in range(len(input)):
			# input_id= all_input_ids[i, : ].unsqueeze(0)
			x  = torch.tensor(input[i] , dtype=torch.long).to(self.device).unsqueeze(0) #shape (1 , N_word)
			t = self.pretrained_model(x, output_attentions=True, return_dict=True)
			attention  = t.attentions[-1][0][0][-10:-1, :]
			output.append(attention)
		return output 
	
	def forward_pretrained_tok(self, input ):
		#input is list of list of list (num chunk , num label , num token )
		#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
		# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
		output = []
		for i in range(len(input)):
			#for chunk 
			# input_id= all_input_ids[i, : ].unsqueeze(0)
			x  = torch.tensor(input[i] , dtype=torch.long).to(self.device) #shape (num label , N_word)
			t = self.pretrained_model(x.unsqueeze(0))[0] #shape (1 , N_word , word dim)
			# print('t shape ', t.shape)
			output.append(t)
		return output 

	# def forward_pretrained_tok(self, input ):
	# 	#input is list of list of list (num chunk , num label , num token )
	# 	#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
	# 	# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
	# 	output = []
	# 	for i in range(len(input)):
	# 		#for chunk 
	# 		# input_id= all_input_ids[i, : ].unsqueeze(0)
	# 		# x  = torch.tensor(input[i] , dtype=torch.long).to(self.device) #shape (num label , N_word)
	# 		# print('x shape ', x.shape )
	# 		chunk_hid = []
	# 		for j in range(len(input[i])):
	# 			index = input[i][j]
	# 			# input_id= index.unsqueeze(0)
	# 			# print(index)
	# 			x  = torch.tensor(index , dtype=torch.long).to(self.device).unsqueeze(0)
	# 			t = self.pretrained_model(x)[0] #shape (1 , N_word , word dim)
	# 			# print('t shape ', t.shape)
	# 			chunk_hid.append(t.squeeze(0)) #(N word , word dim)
	# 		chunk_hid = torch.stack(chunk_hid) # N label , N word , word dim
	# 		output.append(chunk_hid)
	# 	return output 
	
	def forward_pretrained_label(self, input ):
		#input shape list of list of indexes of tokens in citation context [ [] , [] ] 
		# all_input_ids = torch.tensor([s.input_ids for s in features], dtype=torch.long).to(self.device)
		x  = torch.tensor(input , dtype=torch.long).to(self.device).unsqueeze(0) #shape (1 , N_word)
		t = self.pretrained_model(x)[0] #shape (1 , N_word , word dim)
		return t.squeeze(0)

	def sent_pool(self, tok_embed, tok2sent_ind ,  method_pool = 'mean'):
		#tok_embed: shape (sequen_length , hidden size)
		bound_sents = find_boudary_sents(tok2sent_ind)
			
		sents_emb = [] 
		for bound_sent in bound_sents:
			sents_emb.append(self.pool_sequential_embed(tok_embed , bound_sent[0] , bound_sent[1] , method_pool) )
		sents_emb = torch.stack(sents_emb , dim = 0) #shape (N_sent , sent dim )

		return sents_emb

	def pool_sequential_embed(self, roberta_embed , start , end , method):

		if method =='mean':
			sub_matrix = roberta_embed[start:end+1 , :] 
			return torch.mean(sub_matrix , axis = 0 ) 
		elif method == 'att':
			#func_emb (N_func , word dim) 
			#using one attention layer to compute the document embedding
			sub_matrix = roberta_embed[start:end+1 , :] 

			func_att = self.W_q_sent(sub_matrix) #shape (N func , 1)
			func_val = self.W_v_sent(sub_matrix) #shape (N func , word dim)

			attention_matrix = torch.nn.functional.softmax(func_att , dim = 0) #shape (N_func , 1 )
			attention_matrix = torch.transpose(attention_matrix , 0 , 1) #shape (1 , N func )
			result = torch.matmul(attention_matrix ,  func_val) #shape (1 , word_dim)
			return result.squeeze(0)

def find_boudary_sents(tok2sent_idx):
	#bound sents
	sent_tok = {}

	for i in range(len(tok2sent_idx)):
		if tok2sent_idx[i] not in sent_tok:
			sent_tok[tok2sent_idx[i]] = [i]
		else:
			sent_tok[tok2sent_idx[i]].append(i)

	bound_sents = [] 
	for sent_id in sent_tok:
		bound_sents.append([sent_tok[sent_id][0],sent_tok[sent_id][-1]])

	return bound_sents
