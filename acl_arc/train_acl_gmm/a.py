# import torch 
# X = torch.tensor([ [1,2,3], [4,5,6] ])
# Y = torch.tensor([ [7, 8, 9], [9 , 10 , 11] , [11, 23, 12] ] )
# X1 = X.unsqueeze(1)
# Y1=Y.unsqueeze(0)
# print(X1.shape,Y1.shape)
# X2 = X1.repeat(1 , Y.shape[0],1)
# Y2 = Y1.repeat(X.shape[0], 1,1)
# print(X2.shape,X2.shape)
# Z = torch.cat([X2,Y2],-1)
# Z = Z.view(-1,Z.shape[-1])
# print(Z.view( 2 , 3 , -1 ))

from sklearn.metrics import f1_score

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 1, 0, 1, 0]

print("micro-F1(a,b):", f1_score(y_true, y_pred, average='micro'))
print("micro-F1(b,a):", f1_score(y_pred, y_true, average='micro'))

print("macro-F1(a,b):", f1_score(y_true, y_pred, average='macro'))
print("macro-F1(b,a):", f1_score(y_pred, y_true, average='macro'))
