import numpy
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext.vocab import Vectors, GloVe
import torch.nn.functional as F
import pdb
from copy import deepcopy
import pdb

'''
## Predecessor
##------DATA--------
num_constants = 10
background_predicates =[0,1]
intensional_predicates=[2]

##Background Knowledge
zero_extension = torch.zeros(1,num_constants).view(-1,1)
zero_extension[0,0] = 1
succ_extension = torch.eye(num_constants-1,num_constants-1)
succ_extension = torch.cat((torch.zeros(num_constants-1,1),succ_extension),1)
succ_extension = torch.cat((succ_extension,torch.zeros(1,num_constants)),0)

#Intensional Predicates
predecessor_extension = torch.zeros(num_constants,num_constants)

##Target
steps = 2
target = Variable(torch.zeros(num_constants,num_constants))
target[1,0] =1
target[2,1] =1
target[3,2] =1
target[4,3] =1
target[5,4] =1
target[6,5] =1
target[7,6] =1
target[8,7] =1
target[9,8] =1


valuation_init = [Variable(zero_extension), Variable(succ_extension), Variable(predecessor_extension)]

num_rules = 1
rules_str = [9]

'''
############ Even/odd

##------DATA--------
num_constants = 5
background_predicates =[0,1]
intensional_predicates=[2,3]

##Background Knowledge
zero_extension = torch.zeros(1,num_constants).view(-1,1)
zero_extension[0,0] = 1
succ_extension = torch.eye(num_constants-1,num_constants-1)
succ_extension = torch.cat((torch.zeros(num_constants-1,1),succ_extension),1)
succ_extension = torch.cat((succ_extension,torch.zeros(1,num_constants)),0)

#Intensional Predicates
aux_extension = torch.zeros(1,num_constants).view(-1,1)
even_extension = torch.zeros(1,num_constants).view(-1,1)

##Target
target = Variable(torch.zeros(1,num_constants)).view(-1,1)
steps = 5
even = [0,2,4]
for integer in even:
    target[integer,0]=1

valuation_init = [Variable(zero_extension), Variable(succ_extension), Variable(aux_extension), Variable(even_extension)]

num_rules = 3
rules_str = [1,2,2]

'''
############### Less-Than
##------DATA--------
num_constants = 10
background_predicates =[0,1]
intensional_predicates=[2]

##Background Knowledge
zero_extension = torch.zeros(1,num_constants).view(-1,1)
zero_extension[0,0] = 1
succ_extension = torch.eye(num_constants-1,num_constants-1)
succ_extension = torch.cat((torch.zeros(num_constants-1,1),succ_extension),1)
succ_extension = torch.cat((succ_extension,torch.zeros(1,num_constants)),0)

#Intensional Predicates
less_extension = torch.zeros(num_constants, num_constants)

##Target
steps = 5
target = Variable(torch.zeros(num_constants, num_constants))
for i in range(num_constants):
    for j in range(i):
        target[j,i] = 1

valuation_init = [Variable(zero_extension), Variable(succ_extension), Variable(less_extension)]


num_rules = 2
rules_str = [5,3]

#################### Son
num_constants = 9
background_predicates =[0,1,2]
intensional_predicates=[3,4]

##Background Knowledge
father_extension = torch.zeros(num_constants,num_constants)
brother_extension = torch.zeros(num_constants,num_constants)
sister_extension = torch.zeros(num_constants,num_constants)

father_extension[0,1] = 1 
father_extension[0,2] = 1 
father_extension[3,4] = 1 
father_extension[3,5] =	1
father_extension[6,7] = 1 
father_extension[6,8] =	1
brother_extension[1,2] = 1 
brother_extension[2,1] = 1
brother_extension[4,5] = 1 
sister_extension[5,4] =	1
sister_extension[7,8] = 1 
sister_extension[8,7] =	1

#Intensional Predicates
aux_extension = torch.zeros(1,num_constants).view(-1,1)
son_extension = torch.zeros(num_constants,num_constants)

##Target
target = Variable(torch.zeros(num_constants, num_constants))
target[1,0] = 1
target[2,0] = 1
target[4,3] = 1

steps = 2

valuation_init = [Variable(father_extension), Variable(brother_extension), Variable(sister_extension), Variable(aux_extension), Variable(son_extension)]

num_rules = 3
rules_str = [11, 12, 12]

############# Husband Requires Dataset       
num_rules = 1
rules_str = [3]

############# Uncle Requires Dataset
num_rules = 3
rules_str = [3,5,5]

####### Relatedness
num_constants = 8
background_predicates =[0]
intensional_predicates=[1,2]

##Background Knowledge
parent_extension = torch.zeros(num_constants,num_constants)

parent_extension[0,1] = 1
parent_extension[0,2] = 1
parent_extension[2,4] = 1 
parent_extension[2,5] = 1
parent_extension[3,2] = 1
parent_extension[6,7] = 1

#Intensional Predicates
aux_extension = torch.zeros(num_constants, num_constants)
related_extension = torch.zeros(num_constants,num_constants)

##Target
target = torch.ones(num_constants-2, num_constants-2)
target = torch.cat((target, torch.zeros(2, num_constants -2)), 0)
target = torch.cat((target, torch.zeros(num_constants,2)), 1)
target[6,7] = 1
target[6,6] = 1
target[7,7] = 1
target[7,6] = 1
target = Variable(target)
steps = 4

valuation_init = [Variable(parent_extension), Variable(aux_extension), Variable(related_extension)]


num_rules = 4
rules_str = [5,3,5,9]

########### Father
num_constants = 12
background_predicates =[0,1,2,3]
intensional_predicates=[4]

##Background Knowledge
brother_extension = torch.zeros(num_constants,num_constants)
husband_extension = torch.zeros(num_constants,num_constants)
mother_extension = torch.zeros(num_constants,num_constants)
aunt_extension = torch.zeros(num_constants,num_constants)

mother_extension[0,1] = 1
brother_extension[2,1] = 1
husband_extension[2,3] = 1
aunt_extension[3,4] = 1
mother_extension[3,5] = 1
aunt_extension[3,6] = 1
brother_extension[7,8] = 1
husband_extension[8,9] = 1
brother_extension[9,10] = 1
mother_extension[9,11] = 1

#Intensional Predicates
father_extension = torch.zeros(num_constants,num_constants)

##Target
target = Variable(torch.zeros(num_constants, num_constants))
target[2,5] = 1
target[8,11] = 1

steps = 1

valuation_init = [Variable(brother_extension), Variable(husband_extension), Variable(mother_extension), Variable(aunt_extension), Variable(father_extension)]

num_rules = 1
rules_str = [3]

############### Graph connectedness 

num_constants = 4
background_predicates =[0]
intensional_predicates=[1]

##Background Knowledge
edge_extension = torch.zeros(num_constants,num_constants)
edge_extension[0,1] = 1
edge_extension[1,2] = 1
edge_extension[2,3] = 1
edge_extension[1,0] = 1

#Intensional Predicates
connected_extension = torch.zeros(num_constants,num_constants)

##Target
target = Variable(torch.zeros(num_constants, num_constants))
target[0,0] = 1
target[0,1] = 1
target[0,2] = 1
target[0,3] = 1
target[1,0] = 1
target[1,1] = 1
target[1,2] = 1
target[1,3] = 1
target[2,3] = 1

steps = 3

valuation_init = [Variable(edge_extension), Variable(connected_extension)]
 
num_rules = 2
rules_str = [5, 3]


background_predicates =[0,1]
intensional_predicates=[2,3]

##Background Knowledge
num_constants = 5

neq_extension = torch.ones(num_constants, num_constants)
for i in range(num_constants):
    neq_extension[i,i] = 0

edge_extension = torch.zeros(num_constants, num_constants)
edge_extension[0,1] = 1
edge_extension[0,2] = 1
edge_extension[1,3] = 1
edge_extension[2,3] = 1
edge_extension[2,4] = 1
edge_extension[3,4] = 1

#Intensional Predicates
target_extension = torch.zeros(1, num_constants).view(-1,1)
aux_extension = torch.zeros(num_constants,num_constants)

valuation_init = [Variable(neq_extension), Variable(edge_extension), Variable(aux_extension), Variable(target_extension)]

##Target
target = Variable(torch.zeros(1,num_constants)).view(-1,1)
target[0,0] = 1
target[2,0] = 1

steps = 2

num_rules = 2
rules_str = [14,3]

'''

## 1 F(x) <-- F(X)
## 2 F(x)<---F(Z),F(Z,X)
## 3 F(x,y)<-- F(x,Z),F(Z,Y)
## 4 F(X) <-- F(X,X)
## 5 F(X,Y) <-- F(X,Y)
## 8 F(X,X) <-- F(X)

##----Num tasks
## 9 F(x,y) <-- F(y,x)
## 10 F(x,y)<---F(y,Z),F(X,Z)
## 11 F(x,y)<-- F(y,x),F(x)
## 12 F(X) <-- F(X,Z)
## 13 F(X) <-- F(X,Z), F(Z)
## 14 F(X) <-- F(X,Z), F(X,Z)


num_predicates= len(valuation_init)
num_intensional_predicates = len(intensional_predicates)
num_feat = num_predicates



##------FORWARD CHAINING------
def decoder_efficient(valuation, step):
    ##Unifications
    rules_aux = torch.cat((noisy_rules[:,:num_feat], noisy_rules[:,num_feat:2*num_feat], noisy_rules[:,2*num_feat:3*num_feat]),0)
    rules_aux = rules_aux.repeat(num_predicates,1)
    embeddings_aux = noisy_embeddings.repeat(1,num_rules*3).view(-1,num_feat)    
    unifs = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)
    #unifs = F.pairwise_distance(embeddings_aux, rules_aux).view(num_predicates,-1)
    #unifs = torch.exp(-unifs)
    #unifs_sum = torch.sum(unifs, 0)
    #unifs= unifs/unifs_sum
    
    ##Get_Valuations
    valuation_new = [valuation[i].clone() for i in background_predicates] + \
                    [Variable(torch.zeros(valuation[i].size())) for i in intensional_predicates] 
    
    for predicate in intensional_predicates:
        for s in range(num_constants):
            if valuation[predicate].size()[1] == 1:
                max_score = Variable(torch.Tensor([0]))
                for rule in range(num_rules):
                    if rules_str[rule] == 1:
                        for body1 in range(num_predicates):
                            if valuation[body1].size()[1] == 1:
                                unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                num = valuation[body1][s,0]
                                score_rule = unif*num
                                max_score = torch.max(max_score, score_rule)
                    elif rules_str[rule] == 2:
                        for body1 in range(num_predicates):
                            if valuation[body1].size()[1] == 1:
                                for body2 in range(num_predicates):
                                    if valuation[body2].size()[1] > 1:
                                        unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                        num = torch.min(valuation[body1][:,0],valuation[body2][:,s])
                                        num = torch.max(num)
                                        score_rule = unif*num
                                        max_score = torch.max(max_score, score_rule)
                    elif rules_str[rule] == 4:
                        for body1 in range(num_predicates):
                            if valuation[body1].size()[1] > 1:
                                unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                num = valuation[body1][s,s]
                                score_rule = unif*num
                                max_score = torch.max(max_score, score_rule)
                    elif rules_str[rule] == 12:
                        for body1 in range(num_predicates):
                            if valuation[body1].size()[1] > 1:
                                unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                num = torch.max(valuation[body1][s,:])
                                score_rule = unif*num
                                max_score = torch.max(max_score, score_rule)
                    elif rules_str[rule] == 13:
                        for body1 in range(num_predicates):
                            if valuation[body1].size()[1] > 1:
                                for body2 in range(num_predicates):
                                    if valuation[body2].size()[1] == 1:
                                        unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                        num = torch.min(valuation[body1][s,:],valuation[body2][:,0])
                                        num = torch.max(num)
                                        score_rule = unif*num
                                        max_score = torch.max(max_score, score_rule)
                    elif rules_str[rule] == 14:
                        for body1 in range(num_predicates):
                            if valuation[body1].size()[1] > 1:
                                for body2 in range(num_predicates):
                                    if valuation[body2].size()[1] > 1:
                                        unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                        num = torch.min(valuation[body1][s,:],valuation[body2][s,:])
                                        num = torch.max(num)
                                        score_rule = unif*num
                                        max_score = torch.max(max_score, score_rule)

                valuation_new[predicate][s,0] = torch.max(valuation[predicate][s,0], max_score)
            else:
                for o in range(num_constants):
                    max_score = Variable(torch.Tensor([0]))
                    for rule in range(num_rules):
                        if rules_str[rule] == 3:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    for body2 in range(num_predicates):
                                        if valuation[body2].size()[1] > 1:
                                            unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                            num = torch.min(valuation[body1][s,:],valuation[body2][:,o])
                                            num = torch.max(num)
                                            score_rule = unif*num
                                            max_score = torch.max(max_score, score_rule)
                        elif rules_str[rule] == 5:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                    num = valuation[body1][s,o]
                                    score_rule = unif*num
                                    max_score = torch.max(max_score, score_rule)
                        elif rules_str[rule] == 8:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] == 1:
                                    unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                    num = valuation[body1][s,0]
                                    score_rule = unif*num
                                    max_score = torch.max(max_score, score_rule)
                        elif rules_str[rule] == 9:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                    num = valuation[body1][o,s]
                                    score_rule = unif*num
                                    max_score = torch.max(max_score, score_rule)
                        elif rules_str[rule] == 10:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    for body2 in range(num_predicates):
                                        if valuation[body2].size()[1] > 1:
                                            unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                            num = torch.min(valuation[body1][s,:],valuation[body2][o,:])
                                            num = torch.max(num)
                                            score_rule = unif*num
                                            max_score = torch.max(max_score, score_rule)
                        elif rules_str[rule] == 11:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    for body2 in range(num_predicates):
                                        if valuation[body2].size()[1] == 1:
                                            unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                            num = torch.min(valuation[body1][o,s],valuation[body2][s,0])
                                            num = torch.max(num)
                                            score_rule = unif*num
                                            max_score = torch.max(max_score, score_rule)


                    valuation_new[predicate][s,o] = torch.max(valuation[predicate][s,o], max_score)              
    return valuation_new
            
def amalgamate(x,y):
    return x + y - x*y

##------SETUP------
num_iters = 1200
learning_rate = .1
learning_rate_rules1=.1
learning_rate_rules2=.05
learning_rate_rules3=.01
hyper_epsilon = .0
hyper_epoch = 30

hyper_epsilon_r = 1.50
hyper_epoch_r = 60




#embeddings = Variable(torch.rand(num_predicates, num_feat), requires_grad=True)
#embeddings = Variable(torch.zeros(num_predicates, num_feat)+.9, requires_grad=True)
embeddings = Variable(torch.eye(num_predicates), requires_grad=True)

rules = Variable(torch.rand(num_rules, num_feat*3), requires_grad=True)
#rules = Variable(torch.zeros(num_rules, num_feat*3)+.1, requires_grad=True)
#rule1 = torch.Tensor([0,0,0,1,1,0,0,0,0,0,1,0]).view(1,-1)
#rule2 = torch.Tensor([0,0,0,1,0,0,1,0,0,1,0,0]).view(1,-1)
#rule3 = torch.Tensor([0,0,1,0,0,0,0,1,0,1,0,0]).view(1,-1)
#rules = Variable(torch.cat((rule1,rule2,rule3),0), requires_grad=True)
optimizer = torch.optim.Adam([

             {'params': [embeddings]},
             {'params': [rules], 'lr': learning_rate_rules1}
             #{'params': [rules[0,:]], 'lr': learning_rate_rules1},
             #{'params': [rules[1,:]], 'lr': learning_rate_rules2},  
             #{'params': [rules[2,:]], 'lr': learning_rate_rules3}
             ],lr=learning_rate)
criterion = torch.nn.BCELoss(size_average=False)

##-------TRAINING------
for epoch in range(num_iters):
    if epoch%hyper_epoch == 0:
        hyper_epsilon = hyper_epsilon/2.
    if epoch%hyper_epoch_r == 0:
        hyper_epsilon_r = hyper_epsilon_r*.7

    epsilon = torch.randn(rules.size())
    noisy_rules = rules + hyper_epsilon_r*epsilon

    epsilon_emb = torch.randn(embeddings.size())
    noisy_embeddings = embeddings + hyper_epsilon*epsilon_emb

    for par in optimizer.param_groups[:]:
        for param in par['params']:
            param.data.clamp_(min=0.0,max=1.0)

    optimizer.zero_grad()
    valuation = valuation_init

    for step in range(steps):
        valuation = decoder_efficient(valuation,step)
        #print('step',step,'valuation3', valuation[3], 'valuation2',valuation[2])

    loss = criterion(valuation[-1],target)
    print(epoch,'lossssssssssssssssssssssssssss',loss.data[0])

    if epoch<num_iters-1:
        loss.backward()
        optimizer.step()

##------PRINT RESULTS------
#print('embeddings', embeddings)
#print( 'rules',rules)
rules_aux = torch.cat((rules[:,:num_feat],rules[:,num_feat:2*num_feat],rules[:,2*num_feat:3*num_feat]),0)
rules_aux = rules_aux.repeat(num_predicates,1)
embeddings_aux = embeddings.repeat(1,num_rules*3).view(-1,num_feat)
unifs = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)
unifs_sum = torch.sum(unifs, 0)
unifs= unifs/unifs_sum

print('unifications',unifs)

print(target)
print('val',valuation[-1])
accu = torch.sum(torch.round(valuation[-1])==target).data[0]
print('accuracy',accu, '/', target.nelement())


