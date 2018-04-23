
# coding: utf-8

# In[1]:

###el bueno
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
                
knowledge_pos = Variable(knowledge_pos.narrow(0, 1, knowledge_pos.size()[0]-1))
knowledge_neg = Variable(knowledge_neg.narrow(0, 1, knowledge_neg.size()[0]-1))

print(knowledge_pos)
print(knowledge_neg)
print(num_feat_facts)
 



# In[ ]:

def read(state):


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
                new_fact = torch.cat((rule[:num_predicates], fact1[num_predicates:num_predicates+num_constants],                                                               fact2[num_predicates+num_constants:-1], p), 0)
                new_facts = torch.cat(( new_facts, new_fact.view(1,-1) ),0)
    _ , index = torch.topk(new_facts[:,-1], K)
    new_facts = torch.index_select(new_facts, 0, index)
    #pdb.set_trace()            
    #new_facts = torch.topk(new_facts, K, dim = new_facts.size()[1])            
    return new_facts
        
            


# In[6]:

num_iters = 1000
learning_rate = .001
steps = 2
hidden_size_decoder = 500
num_rules = 2

K = 34 ##For top K

#Core Relations:
## Tree of animals---6 ground facts:
#canary, eagle are birds. shark salmon are fishs. fishs,birds are animals
## 1 has_is per object---7 ground facts:
## animal breathes, bird flies, fish swims, canary sings, eagle claws, shark bites, salmon pink

core_rel = Variable(torch.rand(13, num_feat_facts), requires_grad=True)

#rules = Variable(torch.rand(num_rules, num_predicates*3), requires_grad=True)


#embeddings = Variable(torch.eye(num_predicates), requires_grad=True)
#embeddings = Variable(torch.rand(num_predicates, num_feat), requires_grad=True)
rule1 = torch.Tensor([1,0,1,0,1,0]).view(1,-1)
rule2 = torch.Tensor([0,1,1,0,0,1]).view(1,-1)
rules = Variable(torch.cat((rule1,rule2),0), requires_grad=True)

optimizer = torch.optim.Adam([
        #{'params': [rules]},
        {'params': [core_rel]}
    ], lr = learning_rate)

criterion = torch.nn.MSELoss(size_average=False)


for epoch in range(num_iters):
    optimizer.zero_grad()
    facts = torch.cat((core_rel, Variable(torch.ones(13).view(13,1))), 1)
    #print('epoch {} before_decoder'.format(epoch), facts)
        
    for step in range(steps):
        facts = forward_step(facts)
        #if epoch % 39 == 0 and epoch>0 :
            #print('epoch {}, step {}'.format(epoch,step+1), valuation)
    
    
    loss = criterion(facts[:,:-1], knowledge_pos)
    print(epoch, 'losssssssssssssssssssss',loss.data[0])
    #pdb.set_trace()
    loss.backward()
    optimizer.step()
    print(core_rel)

#data[0,6,0],data[1,6,0],data[1,6,2],data[1,6,6] = 0,0,0,0
# Knows: salmon is fish, salmon is salmon




# In[ ]:

print(embeddings)
print(rules)

rules_aux = torch.cat((rules[:,:num_feat],rules[:,num_feat:2*num_feat],rules[:,2*num_feat:3*num_feat]),0)
rules_aux = rules_aux.repeat(num_predicates,1)
embeddings_aux = embeddings.repeat(1,num_rules*3).view(-1,num_feat)


unifs_real = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)
print('aaaaa',unifs_real)
#print('oooo',F.pairwise_distance(embeddings_aux, rules_aux).view(num_predicates,-1))

unifs = F.pairwise_distance(embeddings_aux, rules_aux).view(num_predicates,-1)
unifs = torch.exp(-unifs)
#print(unifs)
unifs_sum = torch.sum(unifs, 0)
unifs= unifs/unifs_sum
print('finaaaal',unifs)


# In[ ]:

rules[0][0]


# In[ ]:

def decoder_efficient(valuation):

    ##Unifications
    rules_aux = torch.cat((rules[:,:num_feat],rules[:,num_feat:2*num_feat],rules[:,2*num_feat:3*num_feat]),0)
    rules_aux = rules_aux.repeat(num_predicates,1)
    embeddings_aux = embeddings.repeat(1,num_rules*3).view(-1,num_feat)
    #embeddings_aux = F.dropout(embeddings_aux, p=.1,training=True)
    #rules_aux = F.dropout(rules_aux, p=.1, training=True)

    unifs = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)
    
    #unifs = F.pairwise_distance(embeddings_aux, rules_aux).view(num_predicates,-1)
    #unifs = torch.exp(-unifs)
    #unifs_sum = torch.sum(unifs, 0)
    #unifs= unifs/unifs_sum
    
    #unifs=Variable(torch.Tensor([[1,0,1,1,1,0],[0,1,0,0,0,1]]))


    valuation_new = Variable(torch.Tensor(valuation.size()))
    for predicate in intensional_predicates:
        for s in range(num_subjects):
            for o in range(num_objects):
                valuation_aux = Variable(torch.Tensor([0]))
                for body1 in range(num_predicates):
                    for body2 in range(num_predicates):
                        num = torch.min(valuation[body1][s,:],valuation[body2][:,o])
                        num = torch.max(num)

                        ## max across three rules
                        new = Variable(torch.Tensor([0]))
                        for rule in range(num_rules): 
                            unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                            new = torch.max(new,unif)
                            #could be amalgamate

                        num = num*new 
                        
                        valuation_aux = amalgamate(valuation_aux, num)
                        
                        #if predicate == 0 and s==1 and o==1 and valuation_aux.data[0]>.3:
                        #    print('body1', body1, 'body2', body2)

                valuation_new[predicate,s,o] = amalgamate(valuation[predicate,s,o], valuation_aux)

    return valuation_new
    
def amalgamate(x,y):
    return x + y - x*y

