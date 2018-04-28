###el bueno
import numpy
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext.vocab import Vectors, GloVe
import torch.nn.functional as F
import pdb
from copy import deepcopy

##DATA

##predicates: is_a, has_a
##subjects: (animal,bird, fish, canary, eagle, shark, salmon)
##objects: (breathes, can fly, can swim, can sing, has claws, can bite, is pink)
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

num_objects = 7
num_subjects = 7
num_constants = num_objects + num_subjects
num_predicates = 2

##one-hot vectors
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
 
_,rules_aux= torch.max(knowledge_pos[:,:2],1)
_,obj_aux= torch.max(knowledge_pos[:,2:16],1)
_,subj_aux= torch.max(knowledge_pos[:,16:],1)
data_aux = torch.cat((rules_aux.view(-1,1), obj_aux.view(-1,1), subj_aux.view(-1,1)),1)

#####There are 20 Core relations: 7 tautologies, 
##7 properties(animals breath, bird flies, fish swims, canary sings, eagle claws, shark bites, salmon pink),
## 6 cores (canary,eagles are birds; shark salmons are fishs; fish,birds are animals)
num_core = 7+7+6

core_indices = torch.LongTensor([0,2,4,7,10,13,16,17,19,21,24,27,30,33,1,3,6,9,12,15])
noncore_indices = torch.LongTensor([5,8,11,14,18,20,22,23,25,26,28,29,31,32])
#knowledge_core = torch.index_select(data_aux, 0, core_indices)
#knowledge_noncore = torch.index_select(data_aux, 0, noncore_indices)
knowledge_core = torch.index_select(knowledge_pos, 0, core_indices)
knowledge_noncore = torch.index_select(knowledge_pos, 0, noncore_indices)
knowledge_order = torch.cat((knowledge_core, knowledge_noncore),0)


####FORWARD CHAINING
def forward_step(facts,drop,training):
    facts = F.dropout(facts, p=drop, training=training)
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
          
                max_prev, indi_prev = torch.max(F.cosine_similarity(new_fact[:-1].view(1,-1).expand(new_facts[:,:-1].size()),new_facts[:,:-1]),0)
                if max_prev.data[0] < 0.9: 
                    new_facts = torch.cat(( new_facts, new_fact.view(1,-1) ),0)
                elif p.data[0] > new_facts[indi_prev.data[0],-1].data[0]:
                    new_facts[indi_prev.data[0]] = new_fact

    _ , index = torch.topk(new_facts[:,-1], K)
    index, _ = torch.sort(index)
    new_facts = torch.index_select(new_facts, 0, index)
    return new_facts
        
####TRAINING
num_iters = 200
learning_rate = .1
lambda_neg = 20.
drop=0

steps = 2
num_rules = 2
epsilon=.001

K = 34 ##For top K

#core_rel = Variable(knowledge_order.narrow(0,0,num_core), requires_grad=True)
core_rel = Variable(torch.rand(num_core, num_feat_facts), requires_grad=True)

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
target_neg = Variable(knowledge_neg)

for epoch in range(num_iters):
    ##For dropout only
    training = True
    if epoch == num_iters-1:
        training= False

    optimizer.zero_grad()
    facts = torch.cat((core_rel, Variable(torch.ones(core_rel.size()[0], 1))), 1)
    for step in range(steps):
        facts = forward_step(facts, drop, training)
    
    ####Separate loss into order core_relation, and not ordered rest of the relations
    loss = criterion(facts[:num_core,:-1], target[:num_core,:])
    #loss = Variable(torch.Tensor([0]))
    for targ in target[num_core:,:]:
	_, indi = torch.max(F.cosine_similarity(targ.view(1,-1).expand(facts[num_core:,:-1].size()),facts[num_core:,:-1]),0)
        indi=indi.data[0]
        loss += criterion(facts[num_core+indi,:-1],targ)/(facts[num_core+indi,-1]+epsilon)

    ##### Negative_loss 
    #loss_neg = Variable(torch.Tensor([0]))
    #for targ in target_neg:
    #    simi, indi = torch.max(F.cosine_similarity(targ.view(1,-1).expand(facts[:,:-1].size()),facts[:,:-1]),0)
    #    if simi.data[0] > .85:
    #        print('neg_simi', indi)
    #        loss_neg += simi
    #loss += lambda_neg*loss_neg

    print(epoch, 'losssssssssssssssssssss',loss.data[0])
    loss.backward()
    optimizer.step()

#### VISUALIZE LEARNED FACTS and RULES
_,rules_aux= torch.max(facts[:,:2],1)
_,obj_aux= torch.max(facts[:,2:16],1)
_,subj_aux= torch.max(facts[:,16:-1],1)
data_aux = torch.cat((rules_aux.view(-1,1), obj_aux.view(-1,1), subj_aux.view(-1,1)),1)
data_aux = torch.cat((data_aux.type(torch.FloatTensor),facts[:,-1].contiguous().view(-1,1)),1)
print('rules', rules)
print('facts',data_aux)

pdb.set_trace()

def amalgamate(x,y):
    return x + y - x*y
