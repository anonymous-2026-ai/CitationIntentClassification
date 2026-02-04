import torch 
import math 

def compute_topics_v3(self, toks_embed , init_emb, return_score = False):
		num_func = init_emb.shape[0]
		#toks emb shape ( num tok ,  word dim )
		out = [] 
		list_attention_matrix = []
		for i in range(num_func):
			sent_query = self.W_q[i](init_emb[i, :].unsqueeze(0)) #shape (1 , tok dim)
			tok_key = self.W_k[i](toks_embed) #shape (N -tok, word dim)
			tok_val = self.W_v[i](toks_embed) #shape (N tok, word dim)
			z1 = ( torch.matmul( sent_query , torch.transpose(tok_key , 0 , 1 )) ) # shape (1 , N tok )
			z = 1. / math.sqrt(self.word_emb_dim) * z1
			attention_matrix = torch.nn.functional.softmax(z , dim = 1) #shape (1 , N_tok )
			result = torch.relu(torch.matmul(attention_matrix ,  tok_val)).squeeze(0) #shape (word dim))
			out.append(result + init_emb[ i , :])
		out = torch.stack(out)
		return out , list_attention_matrix
