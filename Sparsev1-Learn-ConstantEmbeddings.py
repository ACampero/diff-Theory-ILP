
# coding: utf-8

# In[1]:

###el bueno
#import scipy.io
import numpy

import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext.vocab import Vectors, GloVe
import torch.nn.functional as F
import pdb
from copy import deepcopy


# In[2]:

##DATA
#predicates: is_a, has_a
#subjects: (animal,bird, fish, canary, eagle, shark, salmon)
#objects: (breathes, can fly, can swim, can sing, has claws, can bite, is pink)
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




#num_predicates = data.size()[0]
#num_subjects = data.size()[1]
#num_objects = data.size()[2]





# In[4]:

num_objects = 7
num_subjects = 7
num_constants = num_objects + num_subjects
num_predicates = 2
threshold = .1

constants = torch.eye(num_constants)
predicates = torch.eye(num_predicates)

num_feat_facts = predicates.size()[1] + 2*constants.size()[1]

knowledge_pos = torch.zeros([1, num_feat_facts])
knowledge_neg = torch.zeros([1, num_feat_facts])

for predicate in range(num_predicates):
    for obj in range(num_objects):
        for subj in range(num_constants):
            fact = torch.cat((predicates[predicate].view(1,-1), constants[obj].view(1,-1), constants[subj].view(1,-1)), 1)
            if (predicate == 0  and subj<7) or (predicate == 1 and subj>=7):
                if data[predicate, obj, subj%7] == 1:
                    knowledge_pos = torch.cat((knowledge_pos, fact) , 0)
                else:
                    knowledge_neg = torch.cat((knowledge_neg, fact) , 0)
                
knowledge_pos = knowledge_pos.narrow(0, 1, knowledge_pos.size()[0]-1)
knowledge_neg = knowledge_neg.narrow(0, 1, knowledge_neg.size()[0]-1)

print(knowledge_pos)
print(knowledge_neg)
print(num_feat_facts)
 
_,rules_aux= torch.max(knowledge_pos[:,:2],1)
_,obj_aux= torch.max(knowledge_pos[:,2:16],1)
_,subj_aux= torch.max(knowledge_pos[:,16:],1)
data_aux = torch.cat((rules_aux.view(-1,1), obj_aux.view(-1,1), subj_aux.view(-1,1)),1)

#####core 7 identities, 7 properties, 6 core_rel
core_indices = torch.LongTensor([0,2,4,7,10,13,16,17,19,21,24,27,30,33,1,3,6,9,12,15])
noncore_indices = torch.LongTensor([5,8,11,14,18,20,22,23,25,26,28,29,31,32])
#print(torch.index_select(aux, 0, torch.LongTensor([0,5])))s
#knowledge_core = torch.index_select(data_aux, 0, core_indices)
#knowledge_noncore = torch.index_select(data_aux, 0, noncore_indices)
knowledge_core = torch.index_select(knowledge_pos, 0, core_indices)
knowledge_noncore = torch.index_select(knowledge_pos, 0, noncore_indices)
knowledge_order = torch.cat((knowledge_core, knowledge_noncore),0)

#print(knowledge_order)


# In[ ]:

#def read(state):


# In[5]:

def forward_step(facts):
    new_facts = facts.clone()
    for rule in rules:
        for fact1 in facts:
            for fact2 in facts:
                p = fact1[-1]*fact2[-1]
                #pdb.set_trace()
                p = p*F.cosine_similarity(rule[num_predicates:2*num_predicates].view(1,-1), fact1[:num_predicates].view(1,-1))
                p = p*F.cosine_similarity(rule[2*num_predicates:3*num_predicates].view(1,-1), fact2[:num_predicates].view(1,-1))
                p = p*F.cosine_similarity(fact1[num_predicates+num_constants:-1].view(1,-1) , fact2[num_predicates:num_predicates+num_constants].view(1,-1))
                new_fact = torch.cat((rule[:num_predicates], fact1[num_predicates:num_predicates+num_constants],\
                                                             fact2[num_predicates+num_constants:-1], p), 0)
                if torch.max(F.cosine_similarity(new_fact.view(1,-1).expand(new_facts.size()),new_facts)).data[0] < 0.85: 
                    new_facts = torch.cat(( new_facts, new_fact.view(1,-1) ),0)
    #pdb.set_trace()
    _ , index = torch.topk(new_facts[:,-1], K)
    index, _ = torch.sort(index)
    new_facts = torch.index_select(new_facts, 0, index)
    #new_facts = torch.topk(new_facts, K, dim = new_facts.size()[1])            
    return new_facts
        
            


# In[6]:

num_iters = 6000
learning_rate = .001
steps = 2
hidden_size_decoder = 500
num_rules = 2
num_core = 7+7+6
epsilon=.001

K = 34 ##For top K

#Core Relations:
## Tree of animals---6 ground facts:
#canary, eagle are birds. shark salmon are fishs. fishs,birds are animals
## 1 has_is per object---7 ground facts:
## animal breathes, bird flies, fish swims, canary sings, eagle claws, shark bites, salmon pink


core_rel = Variable(torch.rand(num_core, num_feat_facts), requires_grad=True)
#core_rel = Variable(knowledge_order.narrow(0,0,num_core), requires_grad=True)

#embeddings = Variable(torch.eye(num_predicates), requires_grad=True)
#embeddings = Variable(torch.rand(num_predicates, num_feat), requires_grad=True)
#rule1 = torch.Tensor([1,0,1,0,1,0]).view(1,-1)
#rule2 = torch.Tensor([0,1,1,0,0,1]).view(1,-1)
#rules = Variable(torch.cat((rule1,rule2),0), requires_grad=True)
rules = Variable(torch.rand(num_rules,3*num_rules), requires_grad=True)

optimizer = torch.optim.Adam([
        {'params': [rules]},
        {'params': [core_rel]}
    ], lr = learning_rate)

criterion = torch.nn.MSELoss(size_average=False)

target = Variable(knowledge_order)
for epoch in range(num_iters):
    #pdb.set_trace()
    optimizer.zero_grad()
    facts = torch.cat((core_rel, Variable(torch.ones(core_rel.size()[0], 1))), 1)
    #print('epoch {} before_decoder'.format(epoch), facts)
    #pdb.set_trace()    
    for step in range(steps):
        facts = forward_step(facts)
        #if epoch % 39 == 0 and epoch>0 :
            #print('epoch {}, step {}'.format(epoch,step+1), valuation)
    
    ####Separate loss into order core_relation, and not ordered rest of the relations
    #pdb.set_trace()
    loss = criterion(facts[:num_core,:-1], target[:num_core,:])
    #loss = Variable(torch.Tensor([0]))
    for targ in target[num_core:,:]:
	_, indi = torch.max(F.cosine_similarity(targ.view(1,-1).expand(facts[num_core:,:-1].size()),facts[num_core:,:-1]),0)
        indi=indi.data[0]
        #print(facts[num_core+indi,-1])
      
        loss += criterion(facts[num_core+indi,:-1],targ)/(facts[num_core+indi,-1]+epsilon) 
    print(epoch, 'losssssssssssssssssssss',loss.data[0])
    #pdb.set_trace()
    loss.backward()
    optimizer.step()
    #print(core_rel)

#data[0,6,0],data[1,6,0],data[1,6,2],data[1,6,6] = 0,0,0,0
# Knows: salmon is fish, salmon is salmon
pdb.set_trace()



# In[ ]:

print(embeddings)
def amalgamate(x,y):
    return x + y - x*y
