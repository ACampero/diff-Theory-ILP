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

num_objects = 7
num_subjects = 7
num_constants = num_objects + num_subjects
num_predicates = 2

##one-hot vectors
#constants = torch.eye(num_constants)
#predicates = torch.eye(num_predicates)

##Embeddings
constants = Variable(torch.rand(14,num_constants), requires_grad=True)
predicates = Variable(torch.rand(2,num_predicates), requires_grad=True)

num_feat_predicates= predicates.size()[1]
num_feat_constants = constants.size()[1]
num_feat_facts = predicates.size()[1] + 2*constants.size()[1]

knowledge_pos = torch.cat((predicates[0].view(1,-1), constants[0].view(1,-1), constants[0].view(1,-1)), 1)   ##will be erased
knowledge_neg = torch.cat((predicates[0].view(1,-1), constants[0].view(1,-1), constants[0].view(1,-1)), 1)        ##will be erased  

for predicate in range(num_predicates):
    for obj in range(num_constants):
        for subj in range(num_constants):
            fact = torch.cat((predicates[predicate].view(1,-1), constants[obj].view(1,-1), constants[subj].view(1,-1)), 1)
            if obj<num_objects and ((predicate == 0  and subj<7) or (predicate == 1 and subj>=7)):
                if data[predicate, obj, subj%7] == 1:
                    knowledge_pos = torch.cat((knowledge_pos, fact) , 0)
                else:
                    knowledge_neg = torch.cat((knowledge_neg, fact) , 0)
            else:
                knowledge_neg = torch.cat((knowledge_neg, fact) , 0)                
knowledge_pos = knowledge_pos.narrow(0, 1, knowledge_pos.size()[0]-1)
knowledge_neg = knowledge_neg.narrow(0, 1, knowledge_neg.size()[0]-1)
 
#####There are 20 Core relations: 7 tautologies, 
##7 properties(animals breath, bird flies, fish swims, canary sings, eagle claws, shark bites, salmon pink),
## 6 cores (canary,eagles are birds; shark salmons are fishs; fish,birds are animals)


core_indices = Variable(torch.LongTensor([0,2,4,7,10,13,16,17,19,21,24,27,30,33,1,3,6,9,12,15]))
noncore_indices = Variable(torch.LongTensor([5,8,11,14,18,20,22,23,25,26,28,29,31,32]))
sparse_core=0

##For Sparse
###CASE1, Knows Salmon is a fish: infers salmon is animal, salmon breaths, salmon swims
##data[0,6,0]-14,data[1,6,0]-31,data[1,6,2]-32  = 0,0,0, knows [0,6,2]  
##core_indices = torch.LongTensor([0,2,4,7,10,13,16,17,19,21,24,27,30,33,1,3,6,9,12,15])
##noncore_indices = torch.LongTensor([5,8,11,18,20,22,23,25,26,28,29])

###CASE2, Knows Salmon swims and breaths, infers salmon is a fish, salmon is an animal
##data[0,6,0]-14, data[0,6,2]-15 = 0,0,0  knows [1,6,2],[1,6,0]
#core_indices = torch.LongTensor([0,2,4,7,10,13,16,17,19,21,24,27,30,33,1,3,6,9,12])
#noncore_indices = torch.LongTensor([5,8,11,18,20,22,23,25,26,28,29,31,32])
#sparse_core = 1

knowledge_core = torch.index_select(knowledge_pos, 0, core_indices)
knowledge_noncore = torch.index_select(knowledge_pos, 0, noncore_indices)
knowledge_order = torch.cat((knowledge_core, knowledge_noncore),0)

num_core = knowledge_core.size()[0] 


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

                #####Condition to add new_fact only if it is not already known.
                new_facts = torch.cat((new_facts,new_fact.view(1,-1)),0)
                _,rules_aux_forw= torch.max(new_facts[:,:num_feat_predicates],1)
                _,obj_aux_forw= torch.max(new_facts[:,num_feat_predicates:num_feat_predicates+num_feat_constants],1)
                _,subj_aux_forw = torch.max(new_facts[:,num_feat_predicates+num_feat_constants:-1],1)
                data_aux_forw = torch.cat((rules_aux_forw.view(-1,1), obj_aux_forw.view(-1,1), subj_aux_forw.view(-1,1)),1)

                equals, indi_prev = torch.max(torch.sum(data_aux_forw[-1].expand(data_aux_forw[:-1,:].size()) == data_aux_forw[:-1,:],1),0)          
                if equals.data[0] == 3:
                    if p.data[0] > new_facts[indi_prev.data[0],-1].data[0]:
                        new_facts[indi_prev.data[0]] = new_fact
                    new_facts = new_facts.narrow(0, 0, new_facts.size()[0]-1)

                #max_prev, indi_prev = torch.max(F.cosine_similarity(new_fact[:-1].view(1,-1).expand(new_facts[:,:-1].size()),new_facts[:,:-1]),0)
                #if max_prev.data[0] < 0.9: 
                #    new_facts = torch.cat(( new_facts, new_fact.view(1,-1) ),0)
                #elif p.data[0] > new_facts[indi_prev.data[0],-1].data[0]:
                #    new_facts[indi_prev.data[0]] = new_fact

    _ , index = torch.topk(new_facts[:,-1], K)
    index, _ = torch.sort(index)
    new_facts = torch.index_select(new_facts, 0, index)
    return new_facts
        
####TRAINING
num_iters = 500
learning_rate = .1
lambda_neg = 0.01
lambda_core = 0.
drop=0

steps = 2
num_rules = 2
epsilon=.001

K = 34 ##For top K

#core_rel = Variable(knowledge_order.narrow(0,0,num_core), requires_grad=True)
core_rel = Variable(torch.rand(num_core+sparse_core, num_feat_facts), requires_grad=True)

#rule1 = torch.Tensor([1,0,1,0,1,0]).view(1,-1)
#rule2 = torch.Tensor([0,1,1,0,0,1]).view(1,-1)
#rules = Variable(torch.cat((rule1,rule2),0), requires_grad=True)
rules = Variable(torch.rand(num_rules,3*num_rules), requires_grad=True)

optimizer = torch.optim.Adam([
        {'params': [rules]},
        {'params': [core_rel]},
        {'params': [constants,predicates]}
    ], lr = learning_rate)

criterion = torch.nn.MSELoss(size_average=False)

target = knowledge_order
target_neg = knowledge_neg
_,rules_aux_neg= torch.max(target_neg[:,:num_feat_predicates],1)
_,obj_aux_neg= torch.max(target_neg[:,num_feat_predicates:num_feat_predicates+num_feat_constants],1)
_,subj_aux_neg= torch.max(target_neg[:,num_feat_predicates+num_feat_constants:],1)
data_aux_neg = torch.cat((rules_aux_neg.view(-1,1), obj_aux_neg.view(-1,1), subj_aux_neg.view(-1,1)),1)

for epoch in range(num_iters):
    ###Generate target
    print('predicate embeddings',predicates)
    print('rules', rules)

    ##For dropout only
    training = True
    if epoch == num_iters-1:
        training= False

    ##Weight Clipping
    for par in optimizer.param_groups:
        for param in par['params']:
            param.data.clamp_(min=0.,max=1.)

    optimizer.zero_grad()

    facts = torch.cat((core_rel, Variable(torch.ones(core_rel.size()[0], 1))), 1)
    for step in range(steps):
        facts = forward_step(facts, drop, training)

    ##### Visualize
    _,rules_aux= torch.max(facts[:,:num_feat_predicates],1)
    _,obj_aux= torch.max(facts[:,num_feat_predicates:num_feat_predicates+num_feat_constants],1)
    _,subj_aux= torch.max(facts[:,num_feat_predicates+num_feat_constants:-1],1)
    data_aux = torch.cat((rules_aux.view(-1,1), obj_aux.view(-1,1), subj_aux.view(-1,1)),1)    


    ####Separate loss into order core_relation, and not ordered rest of the relations
    loss = F.mse_loss(facts[:num_core,:-1], target[:num_core,:],size_average=False)
    #loss = Variable(torch.Tensor([0]))

    ###For Sparse case, just make sure it doesnt repeat:
    for sparse_indi in range(sparse_core):
        equals, facts_indi = torch.max(torch.sum(data_aux[num_core+sparse_indi].expand(data_aux[:num_core,:].size()) == data_aux[:num_core,:],1),0)
        if equals.data[0] == 3:
            print("falle")
            simi = 1-F.mse_loss(facts[num_core+sparse_indi,:-1].view(1,-1),facts[facts_indi.data[0],:-1].view(1,-1))
            simi = torch.max(simi,Variable(torch.Tensor([0])))
            loss += lambda_core*simi

    ###For extra relations
    for targ in target[num_core:,:]:
	_, indi = torch.max(F.cosine_similarity(targ.view(1,-1).expand(facts[num_core+sparse_core:,:-1].size()),facts[num_core+sparse_core:,:-1]),0)
        indi=indi.data[0]
        loss += F.mse_loss(facts[num_core+sparse_core+indi,:-1],targ,size_average=False)/(facts[num_core+sparse_core+indi,-1]+epsilon)

    ##### Negative_loss 
    loss_neg = Variable(torch.Tensor([0]))
    for targ_indi in range(target_neg.size()[0]):
        equals, facts_indi = torch.max(torch.sum(data_aux_neg[targ_indi].expand(data_aux.size()) == data_aux,1),0)
        if equals.data[0] == 3:
            simi = 1-F.mse_loss(target_neg[targ_indi,:].view(1,-1),facts[facts_indi.data[0],:-1].view(1,-1))
            print('neg_simi', targ_indi, facts_indi.data[0])
            simi = torch.max(simi,Variable(torch.Tensor([0])))  
            loss_neg += simi/(1-facts[facts_indi.data[0],-1]+epsilon)
    loss += lambda_neg*loss_neg

    print(epoch, 'losssssssssssssssssssss',loss.data[0])
    loss.backward(retain_graph=True)
    optimizer.step()

#### VISUALIZE LEARNED FACTS and RULES
_,rules_aux= torch.max(facts[:,:num_feat_predicates],1)
_,obj_aux= torch.max(facts[:,num_feat_predicates:num_feat_predicates+num_feat_constants],1)
_,subj_aux= torch.max(facts[:,num_feat_predicates+num_feat_constants:-1],1)
data_aux = torch.cat((rules_aux.view(-1,1), obj_aux.view(-1,1), subj_aux.view(-1,1)),1)
data_aux = torch.cat((data_aux.type(torch.FloatTensor),facts[:,-1].contiguous().view(-1,1)),1)
print('rules', rules, torch.round(rules))
print('facts',data_aux)

pdb.set_trace()

def amalgamate(x,y):
    return x + y - x*y
