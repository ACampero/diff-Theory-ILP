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


##------DATA--------
num_constants = 7
background_predicates =[0,1]
intensional_predicates=[2,3]

##Background Knowledge
zero_extension = torch.zeros(1,num_constants).view(-1,1)
zero_extension[0,0] = 1
succ_extension = torch.eye(num_constants-1,num_constants-1)
succ_extension = torch.cat((torch.zeros(num_constants-1,1),succ_extension),1)
succ_extension = torch.cat((succ_extension,torch.zeros(1,num_constants)),0)

#Intensional Predicates
aux_extension = torch.zeros(num_constants,num_constants)
even_extension = torch.zeros(1,num_constants).view(-1,1)

##Target
target = Variable(torch.zeros(1,num_constants)).view(-1,1)
steps = 4
even = [0,2,4,6]
for integer in even:
    target[integer,0]=1

num_rules = 3
rules_str = [1,2,3]



## 1 F(x) <-- F(X)
## 2 F(x)<---F(Z),F(Z,X)
## 3 F(x,y)<-- F(x,Z),F(Z,Y)
## 4 F(X) <-- F(X,X)
## 5 F(X,Y) <-- F(X,Y)
## 8 F(X,X) <-- F(X)





##Valuation
valuation_init = [Variable(zero_extension), Variable(succ_extension), Variable(aux_extension), Variable(even_extension)]
num_predicates= len(valuation_init)
num_intensional_predicates = len(intensional_predicates)
num_feat = num_predicates

##------FORWARD CHAINING------
def decoder_efficient(valuation, step):
    ##Unifications
    rules_aux = torch.cat((rules[:,:num_feat],rules[:,num_feat:2*num_feat],rules[:,2*num_feat:3*num_feat]),0)
    rules_aux = rules_aux.repeat(num_predicates,1)
    embeddings_aux = embeddings.repeat(1,num_rules*3).view(-1,num_feat)    
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
                                max_score = amalgamate(max_score, score_rule)
                    elif rules_str[rule] == 2:
                        for body1 in range(num_predicates):
                            if valuation[body1].size()[1] == 1:
                                for body2 in range(num_predicates):
                                    if valuation[body2].size()[1] > 1:
                                        unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                        num = torch.min(valuation[body1][:,0],valuation[body2][:,s])
                                        num = torch.max(num)
                                        score_rule = unif*num
                                        max_score = amalgamate(max_score, score_rule)
                    elif rules_str[rule] == 4:
                        for body1 in range(num_predicates):
                            if valuation[body1].size()[1] > 1:
                                unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                num = valuation[body1][s,s]
                                score_rule = unif*num
                                max_score = amalgamate(max_score, score_rule)
                valuation_new[predicate][s,0] = amalgamate(valuation[predicate][s,0], max_score)
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
                                            max_score = amalgamate(max_score, score_rule)
                        elif rules_str[rule] == 5:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                    num = valuation[body1][s,o]
                                    score_rule = unif*num
                                    max_score = amalgamate(max_score, score_rule)
                        elif rules_str[rule] == 8:
                            for body1 in range(num_predicates):
                                if valuation[body1].size()[1] == 1:
                                    unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                                    num = valuation[body1][s,0]
                                    score_rule = unif*num
                                    max_score = amalgamate(max_score, score_rule)
                    valuation_new[predicate][s,o] = amalgamate(valuation[predicate][s,o], max_score)              
    return valuation_new
            
def amalgamate(x,y):
    return x + y - x*y

##------SETUP------
num_iters = 100
learning_rate = .1


#embeddings = Variable(torch.rand(num_predicates, num_feat), requires_grad=True)
embeddings = Variable(torch.eye(4), requires_grad=True)

rules = Variable(torch.rand(num_rules, num_feat*3), requires_grad=True)
#rule1 = torch.Tensor([0,0,0,1,1,0,0,0,0,1,0,0]).view(1,-1)
#rule2 = torch.Tensor([0,0,0,1,0,0,0,1,0,0,1,0]).view(1,-1)
#rule3 = torch.Tensor([0,0,1,0,0,1,0,0,0,1,0,0]).view(1,-1)
#rules = Variable(torch.cat((rule1,rule2,rule3),0), requires_grad=True)

optimizer = torch.optim.Adam([

             embeddings, 
             rules],lr=learning_rate)
criterion = torch.nn.BCELoss(size_average=False)

##-------TRAINING------
for epoch in range(num_iters):
    for par in optimizer.param_groups[:]:
        for param in par['params']:
            param.data.clamp_(min=0.,max=1.)

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
print('unifications',unifs)

print(target)
print('val',valuation[-1])
accu = torch.sum(torch.round(valuation[-1])==target).data[0]
print('accuracy',accu, '/', target.nelement())

#pdb.set_trace()
