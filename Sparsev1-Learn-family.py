import numpy
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext.vocab import Vectors, GloVe
import torch.nn.functional as F
import pdb
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lamb', default= .1 ,type=float)
parser.add_argument('--sumand', default= .1 ,type=float)
parser.add_argument('--learning_rate_rules', default= .1 ,type=float)
parser.add_argument('--learning_rate', default= .1 ,type=float)
parser.add_argument('--num_iters', default= 300, type=int)

args = parser.parse_args()
##DATA
#predicates: female, spouse, child, mother, father, daugether, son, wife , husband
#constants: a, b, c, d, e, f, g, h, i, j
#           0, 1, 2, 3, 4, 5, 6, 7, 8, 9  

##mother
data_mother = Variable(torch.zeros(10,10))
data_mother[0,2], data_mother[0,3] = 1,1   #a
data_mother[4,1], data_mother[4,6] = 1,1   #e
data_mother[7,8], data_mother[7,9] = 1,1   #h

##father
data_father = Variable(torch.zeros(10,10))
data_father[5,1], data_mother[5,6] = 1,1   #f
data_father[1,2], data_mother[1,3] = 1,1   #b
data_father[6,8], data_mother[6,9] = 1,1   #g

##daughter
data_daughter = Variable(torch.zeros(10,10))
data_daughter[3,0], data_daughter[3,1] = 1,1   #d
data_daughter[9,6], data_daughter[9,7] = 1,1   #j

##son
data_son = Variable(torch.zeros(10,10))
data_son[1,4], data_son[1,5] = 1,1   #b
data_son[6,4], data_son[6,5] = 1,1   #g
data_son[2,0], data_son[2,1] = 1,1   #c
data_son[8,6], data_son[8,7] = 1,1   #i

##wife
data_wife = Variable(torch.zeros(10,10))
data_wife[4,5] =  1   #e
data_wife[0,1] =  1   #a
data_wife[7,6] =  1   #h

##husband
data_husband = Variable(torch.zeros(10,10))
data_husband[5,4] =  1   #f
data_husband[1,0] =  1   #b
data_husband[6,7] =  1   #g


def decoder_efficient(valuation):
    ##Unifications
    rules_aux = torch.cat((rules[:,:num_feat],rules[:,num_feat:2*num_feat],rules[:,2*num_feat:3*num_feat]),0)
    rules_aux = rules_aux.repeat(num_predicates,1)
    embeddings_aux = embeddings.repeat(1,num_rules*3).view(-1,num_feat)
    unifs = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)

    ##Rule Forms
    valuation_new = [Variable(torch.zeros(valuation[i].size())) for i in range(num_predicates)]
    for predicate in range(num_predicates):
        for s in range(num_constants):
            for o in range(valuation[predicate].size()[1]):
                max_score = Variable(torch.Tensor([0]))
                for rule in range(num_rules): 
                    if rules_str[rule] == 0:
                        for body1 in core_predicates[:2]:
                            for body2 in core_predicates[2:]: #0,1 cant be part of body 2
                                unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                num = torch.min(valuation[body1][s,0],valuation[body2][s,o])
                                score_rule = unif*num
                                max_score = amalgamate(max_score, score_rule)
                    elif rules_str[rule] == 1:
                        for body1 in core_predicates[:2]:
                            for body2 in core_predicates[2:]:
                                unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                num = torch.min(valuation[body1][s,0],valuation[body2][o,s])
                                score_rule = unif*num
                                max_score = amalgamate(max_score, score_rule)
                    #else:
                    #    for body1 in core_predicates[:2]:
                    #        unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]
                    #        num = 1-valuation[body1][s,0]
                    #        score_rule = unif*num
                    #        max_score = amalgamate(max_score, score_rule)
                valuation_new[predicate][s,o] = amalgamate(valuation[predicate][s,o], max_score)  ##use amalgamate***+
    return valuation_new
    
def amalgamate(x,y):
    return x + y - x*y

num_iters = args.num_iters
learning_rate = args.learning_rate
learning_rate_rules = args.learning_rate_rules
sumand = args.sumand
steps = 1

num_feat = 10
num_rules = 6
rules_str = [1,1,0,0,0,0]
## 0 = F(x,y)<---F(x),F(x,y)
## 1 = F(x,y)<-- F(x),F(y,x)
## 2 = F(x)<-- 1-F(x)
observed_predicates = [4,5,6,7,8,9] 
core_predicates = [0,1,2,3] 
num_predicates = 10
num_constants = 10

#embeddings = Variable(torch.eye(num_predicates), requires_grad=True)
embeddings = Variable(torch.rand(num_predicates, num_feat), requires_grad=True)

rules = Variable(torch.rand(num_rules, num_feat*3), requires_grad=True)

##female, spouse, child
female = Variable(torch.rand(1,10).view(-1,1), requires_grad=True)
male = Variable(torch.rand(1,10).view(-1,1), requires_grad=True)
spouse = Variable(torch.rand(10,10), requires_grad=True)
child = Variable(torch.rand(10,10), requires_grad=True)

initial_val = [female, male, spouse, child] +[Variable(torch.zeros(10,10)) for i in observed_predicates]

optimizer = torch.optim.Adam([
        {'params': initial_val[:len(core_predicates)]},
        {'params': [rules], 'lr': learning_rate_rules},
        {'params': [embeddings], 'lr':learning_rate_rules},
    ], lr = learning_rate)

criterion = torch.nn.BCELoss(size_average=False)

target = [data_mother, data_father, data_daughter, data_son, data_wife, data_husband]

beta = 1.
lamb= args.lamb

for epoch in range(num_iters):
    for par in optimizer.param_groups[:]:
        for param in par['params']:
            param.data.clamp_(min=0.,max=1.)
            
    optimizer.zero_grad()
    
    valuation = []
    loss_reg = Variable(torch.Tensor([0]))
    for predicate in range(num_predicates):
        valuation = valuation + [initial_val[predicate]] #/(initial_val[predicate]+sumand)]
        loss_reg += torch.sum(valuation[predicate]) 
      
    if epoch % 49 == 0 and epoch>0 :
        print('epoch {} before_decoder'.format(epoch), [torch.round(100*valuation[i])/100. for i in core_predicates])
    for step in range(steps):
        valuation = decoder_efficient(valuation)
        #if epoch % 49 == 0 and epoch>0 :
        #    print('epoch {}, step {}'.format(epoch,step+1), [torch.round(100*valuation[i])/100. for i in range(4)])
    #print(valuation[0],valuation[4])
    loss = Variable(torch.Tensor([0]))
    for predicate in observed_predicates:
        loss += criterion(valuation[predicate],target[predicate - len(core_predicates)]) 
    loss_pos = loss.clone()
    loss = beta*loss
    loss += lamb*loss_reg 
    
    print(epoch, 'losssssssssssssssssssss',torch.round(10000*loss).data[0]/10000,'loss_pos', torch.round(10000*loss_pos).data[0]/10000, 'loss_reg', torch.round(10000*loss_reg).data[0]/10000)
    if epoch<num_iters-1:
        loss.backward()
        optimizer.step()

#print(valuation)
print('embeddings', torch.round(100*embeddings)/100.)
#print('rules', torch.round(100*rules)/100.)

rules_aux = torch.cat((rules[:,:num_feat],rules[:,num_feat:2*num_feat],rules[:,2*num_feat:3*num_feat]),0)
rules_aux = rules_aux.repeat(num_predicates,1)
embeddings_aux = embeddings.repeat(1,num_rules*3).view(-1,num_feat)
unifs_real = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)
#####Unifs real is of the form
#####predicates * [Head1 Head2 Body1 Body21 Body2 Body22], so i normalize in dim,0  
print('unifications',unifs_real)



accu_plus= [torch.sum(torch.clamp(torch.round(valuation[i]+.40)-target[i-4],min=0)) for i in observed_predicates]
print('accu_plus',sum(accu_plus))
accu_neg = [torch.sum(torch.clamp(target[i-4]-torch.round(valuation[i]+.40),min=0)) for i in observed_predicates]
print('accu_neg', sum(accu_neg))
compress = [torch.sum(torch.round(initial_val[i]+.40)).data[0] for i in core_predicates]
print('compress',sum(compress))




pdb.set_trace()

rules[0][0]

