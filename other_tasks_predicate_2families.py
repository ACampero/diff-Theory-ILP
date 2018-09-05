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
###########################Cyclic
##------DATA--------
background_predicates =[0]
intensional_predicates=[1,2]

num_constants = 6
##Background Knowledge
edge_extension = torch.zeros(num_constants, num_constants)
edge_extension[0,1] = 1
edge_extension[1,2] = 1
edge_extension[1,3] = 1
edge_extension[2,0] = 1
edge_extension[3,4] = 1
edge_extension[3,5] = 1
edge_extension[4,5] = 1
edge_extension[5,4] = 1

num_constants_2=5
edge_extension_2 = torch.zeros(num_constants_2, num_constants_2)
edge_extension_2[0,1] = 1
edge_extension_2[2,3] = 1
edge_extension_2[3,4] = 1
edge_extension_2[4,2] = 1

#Intensional Predicates
aux_extension = torch.zeros(num_constants,num_constants)
target_extension = torch.zeros(1,num_constants).view(-1,1)

valuation_init = [Variable(edge_extension), Variable(aux_extension), Variable(target_extension)]

##Target
target = Variable(torch.zeros(1,num_constants)).view(-1,1)
target_aux = [0,1,2,4,5] #Family1
#target_aux = [2,3,4]
for integer in target_aux:
    target[integer,0]=1


#Intensional Predicates
aux_extension_2 = torch.zeros(num_constants_2,num_constants_2)
target_extension_2 = torch.zeros(1,num_constants_2).view(-1,1)

valuation_init_2 = [Variable(edge_extension_2), Variable(aux_extension_2), Variable(target_extension_2)]

##Target
steps = 6
target_2 = Variable(torch.zeros(1,num_constants_2)).view(-1,1)
#target_aux = [0,1,2,4,5] #Family1
target_aux_2 = [2,3,4]
for integer in target_aux_2:
    target_2[integer,0]=1

num_rules = 3
rules_str = [4,5,3]

################### Member
##------DATA--------
background_predicates = [0,1]
intensional_predicates=[2]

num_constants = 5
const_extension = torch.zeros(num_constants, num_constants)
const_extension[4,3] = 1
const_extension[3,2] = 1
const_extension[2,1] = 1
const_extension[1,0] = 1

##Background Knowledge
value_extension = torch.zeros(num_constants, num_constants)
value_extension[0,0] = 1
value_extension[1,1] = 1
value_extension[2,2] = 1
value_extension[3,3] = 1
value_extension[4,4] = 1

num_constants_2 = 5
value_extension_2 = torch.zeros(num_constants_2, num_constants_2)
value_extension_2[0,0] = 1
value_extension_2[1,4] = 1
value_extension_2[2,2] = 1
value_extension_2[3,3] = 1
value_extension_2[4,2] = 1

#Intensional Predicates
target_extension = torch.zeros(num_constants, num_constants)

valuation_init = [Variable(const_extension), Variable(value_extension), Variable(target_extension)]

##Target
target = Variable(torch.zeros(num_constants,num_constants))
for i in range(num_constants):
    for j in range(i+1):
        target[j,i] = 1

#Intensional Predicates
target_extension_2 = torch.zeros(num_constants_2,num_constants_2)

valuation_init_2 = [Variable(const_extension), Variable(value_extension_2), Variable(target_extension_2)]

##Target
target_2 = Variable(torch.zeros(num_constants_2, num_constants_2))           
target_2[0,0] = 1
target_2[0,1] = 1
target_2[0,2] = 1
target_2[0,3] = 1
target_2[0,4] = 1
target_2[4,1] = 1
target_2[4,2] = 1
target_2[4,3] = 1
target_2[4,4] = 1
target_2[2,2] = 1
target_2[2,3] = 1
target_2[2,4] = 1
target_2[3,3] = 1
target_2[3,4] = 1

steps = 5

num_rules = 2
rules_str = [9, 10]


######### Length
##------DATA--------
background_predicates =[0,1,2,3]
intensional_predicates=[4,5]

num_constants = 4
zero_extension = torch.zeros(1,num_constants).view(-1,1)
zero_extension[0,0] = 1
succ_extension = torch.eye(num_constants-1,num_constants-1)
succ_extension = torch.cat((torch.zeros(num_constants-1,1),succ_extension),1)
succ_extension = torch.cat((succ_extension,torch.zeros(1,num_constants)),0)
const_extension = torch.zeros(num_constants, num_constants)
const_extension[3,2] = 1
const_extension[2,1] = 1
const_extension[1,0] = 1

##Background Knowledge
value_extension = torch.zeros(num_constants, num_constants)
value_extension[0,0] = 1
value_extension[1,3] = 1
value_extension[2,2] = 1
value_extension[3,1] = 1

num_constants_2 = 4
value_extension_2 = torch.zeros(num_constants_2, num_constants_2)
value_extension_2[0,0] = 1
value_extension_2[1,1] = 1
value_extension_2[2,2] = 1
value_extension_2[3,1] = 1

#Intensional Predicates
aux_extension = torch.zeros(num_constants,num_constants)
target_extension = torch.zeros(num_constants, num_constants)

valuation_init = [Variable(zero_extension), Variable(succ_extension), Variable(const_extension), Variable(value_extension), Variable(aux_extension), Variable(target_extension)]


##Target
target = Variable(torch.zeros(num_constants,num_constants))
target[0,0] = 1
target[1,1] = 1
target[2,2] = 1
target[3,3] = 1

#Intensional Predicates
aux_extension_2 = torch.zeros(num_constants_2,num_constants_2)
target_extension_2 = torch.zeros(num_constants_2,num_constants_2)

valuation_init_2 = [Variable(zero_extension), Variable(succ_extension), Variable(const_extension), Variable(value_extension_2), Variable(aux_extension_2), Variable(target_extension_2)]

##Target
target_2 = Variable(torch.zeros(num_constants_2,num_constants_2))
target_2[0,0] = 1
target_2[1,1] = 1
target_2[2,2] = 1
target_2[3,3] = 1

steps = 7
num_rules = 3
rules_str = [8,3,3]


################## Undirected Edge
##------DATA--------
background_predicates =[0]
intensional_predicates=[1]

##Background Knowledge
num_constants = 4
edge_extension = torch.zeros(num_constants, num_constants)
edge_extension[0,1] = 1
edge_extension[1,3] = 1
edge_extension[2,2] = 1

num_constants_2 = 4
edge_extension_2 = torch.zeros(num_constants_2, num_constants_2)
edge_extension_2[0,1] = 1
edge_extension_2[2,3] = 1

#Intensional Predicates
target_extension = torch.zeros(num_constants, num_constants)

valuation_init = [Variable(edge_extension), Variable(target_extension)]

##Target
target = Variable(torch.zeros(num_constants,num_constants))
target[0,1] = 1
target[1,0] = 1
target[1,3] = 1
target[3,1] = 1
target[2,2] = 1

#Intensional Predicates
target_extension_2 = torch.zeros(num_constants_2,num_constants_2)

valuation_init_2 = [Variable(edge_extension_2), Variable(target_extension_2)]

##Target
target_2 = Variable(torch.zeros(num_constants_2,num_constants_2))
target_2[0,1] = 1
target_2[1,0] = 1
target_2[2,3] = 1
target_2[3,2] = 1

steps = 1

num_rules = 2
rules_str = [5,9]


################## Adjacent to red
background_predicates =[0,1,2]
intensional_predicates=[3,4]

##Background Knowledge
num_constants = 7
green_extension = torch.zeros(1, num_constants).view(-1,1)
green_extension[5,0] = 1
red_extension = torch.zeros(1, num_constants).view(-1,1)
red_extension[6,0] = 1

edge_extension = torch.zeros(num_constants, num_constants)
color_extension = torch.zeros(num_constants, num_constants)

edge_extension[0,1] = 1
edge_extension[1,0] = 1
edge_extension[2,3] = 1
edge_extension[2,4] = 1
edge_extension[3,4] = 1

color_extension[0,6] = 1
color_extension[1,5] = 1
color_extension[2,6] = 1
color_extension[3,6] = 1
color_extension[5,5] = 1

num_constants_2 = 7
edge_extension_2 = torch.zeros(num_constants_2, num_constants_2)
color_extension_2 = torch.zeros(num_constants_2, num_constants_2)

edge_extension_2[1,2] = 1
edge_extension_2[3,2] = 1

color_extension_2[0,6] = 1
color_extension_2[1,5] = 1
color_extension_2[2,6] = 1
color_extension_2[3,6] = 1
color_extension_2[5,5] = 1

#Intensional Predicates
aux_extension = torch.zeros(1, num_constants).view(-1,1)
target_extension = torch.zeros(1, num_constants).view(-1,1)

valuation_init = [Variable(red_extension), Variable(edge_extension), Variable(color_extension), Variable(aux_extension), Variable(target_extension)]

##Target
target = Variable(torch.zeros(1,num_constants)).view(-1,1)
target[1,0] = 1
target[2,0] = 1

#Intensional Predicates
aux_extension_2 = torch.zeros(1,num_constants_2).view(-1,1)
target_extension_2 = torch.zeros(1,num_constants_2).view(-1,1)

valuation_init_2 = [Variable(red_extension), Variable(edge_extension_2), Variable(color_extension_2), Variable(aux_extension_2), Variable(target_extension_2)]

##Target
target_2 = Variable(torch.zeros(1,num_constants_2)).view(-1,1)
target_2[1,0] = 1
target_2[3,0] = 1

steps = 2
 
num_rules = 2
rules_str = [13, 13]
'''
#####################  two children 
##------DATA--------
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

num_constants_2 = 5
edge_extension_2 = torch.zeros(num_constants_2, num_constants_2)
edge_extension_2[0,1] = 1
edge_extension_2[1,2] = 1
edge_extension_2[1,3] = 1
edge_extension_2[2,2] = 1
edge_extension_2[2,4] = 1
edge_extension_2[3,4] = 1

#Intensional Predicates
target_extension = torch.zeros(num_constants, num_constants)
aux_extension = torch.zeros(num_constants,num_constants)

valuation_init = [Variable(neq_extension), Variable(edge_extension), Variable(aux_extension), Variable(target_extension)]

##Target
target = Variable(torch.zeros(num_constants,num_constants))
target[0,0] = 1
target[2,2] = 1

#Intensional Predicates
target_extension_2 = torch.zeros(num_constants_2, num_constants_2)
aux_extension_2 = torch.zeros(num_constants_2, num_constants_2)

valuation_init_2 = [Variable(neq_extension), Variable(edge_extension_2), Variable(aux_extension_2), Variable(target_extension_2)]

##Target
target_2 = Variable(torch.zeros(num_constants_2, num_constants_2))
target_2[1,1] = 1
target_2[2,2] = 1

steps = 2

num_rules = 2
rules_str = [15,3]

'''
######### Coulouring
background_predicates =[0,1]
intensional_predicates=[2,3]

##Background Knowledge
num_constants = 8
edge_extension = torch.zeros(num_constants, num_constants)
color_extension = torch.zeros(num_constants, num_constants)

edge_extension[0,1] = 1
edge_extension[1,2] = 1
edge_extension[1,3] = 1
edge_extension[2,4] = 1
edge_extension[4,5] = 1

color_extension[0,6] = 1
color_extension[1,7] = 1
color_extension[2,6] = 1
color_extension[3,6] = 1
color_extension[4,7] = 1
color_extension[5,7] = 1

num_constants_2 = 8
edge_extension_2 = torch.zeros(num_constants_2, num_constants_2)
color_extension_2 = torch.zeros(num_constants_2, num_constants_2)

edge_extension_2[0,1] = 1
edge_extension_2[1,0] = 1
edge_extension_2[2,1] = 1
edge_extension_2[5,2] = 1
edge_extension_2[3,2] = 1
edge_extension_2[4,3] = 1

color_extension_2[0,6] = 1
color_extension_2[1,6] = 1
color_extension_2[2,7] = 1
color_extension_2[3,6] = 1
color_extension_2[4,6] = 1
color_extension_2[5,7] = 1


#Intensional Predicates
aux_extension = torch.zeros(num_constants, num_constants)
target_extension = torch.zeros(num_constants, num_constants)

valuation_init = [Variable(edge_extension), Variable(color_extension), Variable(aux_extension), Variable(target_extension)]

##Target
target = Variable(torch.zeros(num_constants,num_constants))
target[4,4] = 1

#Intensional Predicates
aux_extension_2 = torch.zeros(num_constants_2,num_constants_2)
target_extension_2 = torch.zeros(num_constants_2,num_constants_2)

#edge_extension_2 = edge_extension
#color_extension_2 = color_extension
#aux_extension_2 = aux_extension
#target_extension_2 = target_extension

valuation_init_2 = [Variable(edge_extension_2), Variable(color_extension_2), Variable(aux_extension_2), Variable(target_extension_2)]

##Target
target_2 = Variable(torch.zeros(num_constants_2, num_constants_2))
target_2[0,0] = 1
target_2[1,1] = 1
target_2[4,4] = 1
target_2[5,5] = 1

#target_2 = target
steps = 2

num_rules = 2
rules_str = [10,15]

'''
## 1 F(x) <-- F(X)
## 2 F(x)<---F(Z),F(Z,X)
## 3 F(x,y)<-- F(x,Z),F(Z,Y)
## 4 F(X) <-- F(X,X)
## 5 F(X,Y) <-- F(X,Y)
## 8 F(X,X) <-- F(X)

##----Num tasks
## 9 F(x,y) <-- F(y,x)
## 10 F(x,y)<---F(X,Z),F(Y,Z)
## 11 F(x,y)<-- F(y,x),F(x)
## 12 F(X) <-- F(X,Z)
## 13 F(X) <-- F(X,Z), F(Z)
## 14 F(X) <-- F(X,Z), F(X,Z)
## 15 F(X,X) <-- F(X,Z), F(X,Z)

num_predicates= len(valuation_init)
num_intensional_predicates = len(intensional_predicates)
num_feat = num_predicates

##------FORWARD CHAINING------
def decoder_efficient(valuation, step, num_constants):
    ##Unifications
    rules_aux = torch.cat((noisy_rules[:,:num_feat],noisy_rules[:,num_feat:2*num_feat],noisy_rules[:,2*num_feat:3*num_feat]),0)
    rules_aux = rules_aux.repeat(num_predicates,1)
    embeddings_aux = noisy_embeddings.repeat(1,num_rules*3).view(-1,num_feat)
    unifs = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)

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
                        elif rules_str[rule] == 8 and s==o:
                            #print ("mistake! rule 8?")
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
                                            num = torch.min(valuation[body1][s,:], valuation[body2][o,:])
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
                        elif rules_str[rule] == 15 and s==o:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    for body2 in range(num_predicates):
                                        if valuation[body2].size()[1] > 1:
                                            unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                            num = torch.min(valuation[body1][s,:], valuation[body2][s,:])
                                            num = torch.max(num)
                                            score_rule = unif*num
                                            max_score = torch.max(max_score, score_rule)
                    valuation_new[predicate][s,o] = torch.max(valuation[predicate][s,o], max_score)
    return valuation_new

def amalgamate(x,y):
    return x + y - x*y


##------SETUP------
num_iters = 1201
learning_rate = .1
hyper_epsilon_r = 1.5
hyper_epoch_r = 60
hyper_decay_r = .8
#embeddings = Variable(torch.rand(num_predicates, num_feat), requires_grad=True)
embeddings = Variable(torch.eye(num_feat), requires_grad=True)

rules = Variable(torch.rand(num_rules, num_feat*3), requires_grad=True)
#rule1 = torch.Tensor([0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0]).view(1,-1)
#rule2 = torch.Tensor([0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0]).view(1,-1)
#rule3 = torch.Tensor([0,1,0,0,1,0,0,1,0]).view(1,-1)
#rules = Variable(torch.cat((rule1,rule2),0), requires_grad=True)

optimizer = torch.optim.Adam([
             embeddings, 
             rules],lr=learning_rate)
criterion = torch.nn.BCELoss(size_average=False)

##-------TRAINING------
#m=torch.distributions.Bernoulli(torch.Tensor(target.size))
for epoch in range(num_iters):
    if epoch%hyper_epoch_r == 0:
	hyper_epsilon_r = hyper_epsilon_r*hyper_decay_r
    epsilon = torch.randn(rules.size())
    noisy_rules = rules + hyper_epsilon_r*epsilon


    noisy_rules = noisy_rules.detach().clamp_(min=0.0,max=1.0)

    noisy_rules = noisy_rules - rules.detach() + rules

    #epsilon_emb = torch.randn(embeddings.size())
    noisy_embeddings = embeddings #+ hyper_epsilon*epsilon_emb


    for par in optimizer.param_groups[:]:
        for param in par['params']:
            param.data.clamp_(min=0.,max=1.)

    optimizer.zero_grad()
    
    if epoch%2==0:
        valuation = valuation_init
        const_aux = num_constants
        target_aux = target
    else:
        valuation = valuation_init_2
        const_aux = num_constants_2
       	target_aux = target_2

    #pdb.set_trace()
    for step in range(steps):
        #pdb.set_trace()
        valuation = decoder_efficient(valuation,step,const_aux)
        #print('step',step,'valuation3', valuation[3], 'valuation2',valuation[2])
    #pdb.set_trace()
    loss = criterion(valuation[-1],target_aux)
    print(epoch,valuation[-1].size()[0],'lossssssssssssssssssssssssssss',loss.data[0])

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
print('unifications',unifs)
#####Unifs real is of the form
#####predicates * [Head1 Head2 Body1 Body21 Body2 Body22], so i normalize in dim,0

print(target)
print('val',valuation[-1])
accu = torch.sum(torch.round(valuation[-1])==target).data[0]
print('accuracy',accu, '/', target.nelement())

#pdb.set_trace()
