
# coding: utf-8

# In[31]:


import scipy.io
import numpy
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext.vocab import Vectors, GloVe
import torch.nn.functional as F
import pdb
from copy import deepcopy
torch.set_printoptions(profile='short')


# In[4]:

##DATA
#(animal,bird, fish, canary, eagle, shark, salmon)
#(breathes, can fly, can swim, can sing, has claws, can bite, is pink)
data = torch.Tensor(2,7,7).zero_()
##is_a
data[0,:,:7] = torch.eye(7)
data[0,:,0] = torch.ones(7,1)
data[0,3,1], data[0,4,1], data[0,5,2], data[0,6,2] = 1,1,1,1 
##has_a
data[1,:,0] = torch.ones(1,7)
data[1,1,1], data[1,2,2] = 1,1
data[1,3:5,1] = torch.ones(2,1) 
data[1,5:,2] = torch.ones(2,1)
data[1,3:,3:] = torch.eye(4)

##For Sparse
#data[0,6,0],data[1,6,0],data[1,6,2],data[1,6,6] = 0,0,0,0  #It knows [0,6,2] ... salmon is fish
#data[0,6,0],data[1,6,0],data[0,6,2],data[1,6,6] = 0,0,0,0  #It knows [1,6,2] ... salmon can swim

#data[0,6,0],data[0,6,2],data[1,6,6] = 0,0,0  #It knows [1,6,2],[1,6,0] ... salmon can swim and breathe

##Auxiliar, very small
#data = torch.Tensor(2,2,2)
#data[0,:,:] = torch.eye(2,2)
#data[0,1,0] = 1
#data[1,:,0]= torch.Tensor([1,1])




num_predicates = data.size()[0]
num_subjects = data.size()[1]
num_objects = data.size()[2]





# In[5]:

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.linear1 = nn.Linear(num_predicates * num_subjects * num_objects, hidden_size)
        self.decode = nn.Linear(hidden_size, num_predicates * num_subjects * num_objects)
    
    def forward(self,t):
        t = t.view(-1)
        z = self.linear1(t)
        z = F.relu(z)
        z = Variable(torch.ones(1,hidden_size))  ###The encoder is not doing anything
        t0 = self.decode(z)
        t0 = F.sigmoid(t0)
        t0 = t0.view(num_predicates, num_subjects, num_objects)
        return t0


# In[38]:

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        r1 = [0,0,0] #isa(x,y)<-- isa(x,z) isa(x,z)
        r2 = [1,0,1] #hasa(x,y)<-- isa(x,z) hasa(z,y)
        self.rules = torch.LongTensor([r1,r2])
            
    def forward(self,t):
        ## max_rules(max_z min (atom1,atom2),rule2)
        ## instead: Sum_rules(Sum_z dot(atom1,atom2),rule2)
        ##this could be done with a bigger matrix
        t_new = Variable(torch.Tensor(t.size()))
        for predicate in range(num_predicates):
            for s in range(num_subjects):
                for o in range(num_objects):
                    #For a particular s,rule r1=rules[0],o,z
                    #num = t[r1[1],s,z]*t[r1[2],z,o]

                    ##Sum across z
                    #num = torch.mm(t[r1[1],s,:],t[r1[2],:,o].transpose(0,1))  a scalar

                    ##Across heads for s,predicate,o
                    new = Variable(torch.Tensor([0]))
                    for rule in self.rules:
                        if rule[0] == predicate:
                            #num = t[rule[1],s,:]*t[rule[2],:,o]
                            num = torch.min(t[rule[1],s,:],t[rule[2],:,o])
                            num = torch.max(num)
                            #num = torch.dot(t[rule[1],s,:], t[rule[2],:,o])
                            #new = torch.max(num,new)
                            new = self.amalgamate(num,new)
                    #idx_r = torch.LongTensor([i for i,x in enumerate(self.rules) if x[0] == predicate])
                    #idx_r = torch.index_select(self.rules,0, idx_r)
                    #print(idx_r[:,1])
                    #t_aux = torch.index_select(t,0,Variable(idx_r[:,1]))
                    #t_aux2 = torch.index_select(t,0,Variable(idx_r[:,2])).transpose(1,2)
                    #print(t_aux.size(),t_aux2.transpose(1,2).size())
                    #new = torch.sum(torch.mm(t_aux[:,s,:], t_aux2[:,:,o].transpose(1,2)))
                    
                    #t_new[predicate,s,o]= torch.max(t[predicate,s,o], new)
                    t_new[predicate,s,o]= self.amalgamate(t[predicate,s,o], new) 
                    

        return t_new
    
    def amalgamate(self,x,y):
        return x + y - x*y


# In[42]:

###Train
torch.set_printoptions(precision=5)
iterations = 100
steps = 2
hidden_size = 1000
learning_rate = .001

encoder_m = encoder()
decoder_m = decoder()
params = encoder_m.parameters()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.Adam(params, lr = learning_rate)
x = Variable(data)
beta = 1./x.nelement()
lamb= 1.5*1./x.nelement()
lamb2 = 1./x.nelement()
print('lamb',lamb)
print('target',x)
for it in range(iterations):
    encoder_m.zero_grad()
    decoder_m.zero_grad()
    t = encoder_m(x)
    #loss_reg = torch.sum(t.ge(0.1).type(torch.FloatTensor))
    
    loss_reg =torch.sum(t)
    
    #t = deepcopy(x)
    #t[1,1,0]=0
    #t[0,3:,0],t[1,1:,0]= torch.zeros(4,1), torch.zeros(6,1)
    if it % 99 == 0 and it>0 :
        print('it {} t0'.format(it), t)
    for step in range(steps):
        t = decoder_m(t)
        if it % 99 == 0 and it>0 :
            print('it {}, step {}'.format(it,step+1), t)
    
    #loss = lamb2*( criterion(t[1,6,0],x[1,6,0]) + criterion(t[0,6,2], x[0,6,2]) + criterion(t[0,6,6], x[0,6,6]) )
    #loss += beta*criterion(t[:,:-1,:-1],x[:,:-1,:-1]) + lamb*loss_reg
    loss = beta*criterion(t,x) + lamb*loss_reg 
    
    
    print(it, loss.data[0])
    loss.backward()
    optimizer.step()

#data[0,6,0],data[1,6,0],data[1,6,2],data[1,6,6] = 0,0,0,0
# Knows: salmon is fish, salmon is salmon



