
# coding: utf-8

# In[10]:

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


# In[11]:

##DATA
#predicates: is_a, is, can, has
#subjects: 15 liv_thn, plant, tree, pine, oak, flower, rose, daisy, animal, bird, robin, canary, fish, sunfish, salmon
#objects_is: 6 living, tall, pretty, green, red, yellow
#objects_can: 5 grow, move, fly, swim, sing
#objects_has: 10 roots, bark, branches, petals, leaves, skin, wings, feathers, gills, scales
##is_a
dataIS_A = torch.eye(15,15)
dataIS_A[:,0] = torch.ones(15,1) #liv_thn
dataIS_A[1:8,1] = torch.ones(7,1) #plant
dataIS_A[2:5,2] = torch.ones(3,1) #tree
dataIS_A[5:8,5] = torch.ones(3,1) #flower
dataIS_A[8:15,8] = torch.ones(7,1) #animal
dataIS_A[9:12,9] = torch.ones(3,1) #bird
dataIS_A[12:15,12] = torch.ones(3,1) #fish

##is
data_is = torch.Tensor(15,6).zero_()
data_is[:,0] = torch.ones(15,1) #living
data_is[2:5,1] = torch.ones(3,1) #tall
data_is[5:8,2] = torch.ones(3,1) #pretty
data_is[3,3] = 1. #green
data_is[6,4], data_is[10,4], data_is[14,4] = 1., 1., 1.  #red
data_is[7,5], data_is[11,5], data_is[13,5] = 1., 1., 1. #yellow


##can
data_can = torch.Tensor(15,6).zero_()
data_can[:,0] = torch.ones(15,1) #grow
data_can[8:,1] = torch.ones(7,1) #move
data_can[9:12,2] = torch.ones(3,1) #fly
data_can[12:15,3] = 1. #swim
data_can[11,4] = 1.  #sing

##has
#objects_has: 10 roots, bark, branches, petals, leaves, skin, wings, feathers, gills, scales
data_has = torch.Tensor(15,10).zero_()
data_has[1:8,0] = torch.ones(7,1) #roots
data_has[2:5,1] = torch.ones(3,1) #bark
data_has[2:5,2] = torch.ones(3,1) #branches
data_has[5:8,3] = torch.ones(3,1) #petals
data_has[5:8,4] = torch.ones(3,1)  #leaves
data_has[4,4] = 1.  #leaves
data_has[8:,5] = torch.ones(7,1) #skin
data_has[9:12,6] = torch.ones(3,1) #wings
data_has[9:12,7] = torch.ones(3,1) #feathers
data_has[12:,8] = torch.ones(3,1) #gills
data_has[12:,9] = torch.ones(3,1) #scales

total_elements = dataIS_A.nelement() + data_is.nelement() + data_can.nelement() + data_has.nelement()

num_predicates = 4
num_subjects = 15


print(dataIS_A, data_is, data_can, data_has)


# In[44]:

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
    #unifs = F.normalize(unifs,dim=0)
    #unifs=Variable(torch.Tensor([[1,0,1,1,1,0],[0,1,0,0,0,1]]))


    #valuation_new = [deepcopy(valuation[0]),deepcopy(valuation[1]),Variable(torch.zeros(valuation[2].size())),Variable(torch.zeros(valuation[3].size()))]
    valuation_new = [Variable(torch.zeros(valuation[0].size())), Variable(torch.zeros(valuation[1].size())),                     Variable(torch.zeros(valuation[2].size())), Variable(torch.zeros(valuation[3].size()))]
    for predicate in intensional_predicates:
        for s in range(num_subjects):
            for o in range(valuation[predicate].size()[1]):
                valuation_aux = Variable(torch.Tensor([0]))
                for body1 in range(num_predicates):
                    body2 = predicate
                    if valuation[body1].size()[1] == valuation[body2].size()[0]:
                    ##This is allowed because of the following:
                    ## In this case the objects of each predicate are different so the above is justified either
                    ## because the objects are actually different types or 
                    ## because we are only iterating for potential constant matchings
                    ## so an alternative would be to use the below argument but have num_constants and have them go 
                    ## from 0 to num_constants, each valuation[pred] would be bigger 
                    
                    #for body2 in range(num_predicates):
                        num = torch.min(valuation[body1][s,:],valuation[body2][:,o])
                        num = torch.max(num)

                        ## max across three rules
                        new = Variable(torch.Tensor([0]))
                        for rule in range(num_rules): 
                            unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                            new = torch.max(new,unif)
                            #could be amalgamate

                        num = num*new 
                        
                        valuation_aux = amalgamate(valuation_aux, num)  ##use amalgamate for work***+
                        
                        #if predicate == 0 and s==1 and o==1 and valuation_aux.data[0]>.3:
                        #    print('body1', body1, 'body2', body2)

                valuation_new[predicate][s,o] = amalgamate(valuation[predicate][s,o], valuation_aux)  ##use amalgamate***+

    return valuation_new
    
def amalgamate(x,y):
    return x + y - x*y


# In[ ]:


num_iters = 50
learning_rate = .01
learning_rate_rules = .01
steps = 2

num_feat = 2
num_rules = 2
intensional_predicates = [0,1,2,3]
num_intensional_predicates = len(intensional_predicates)

#embeddings = Variable(torch.eye(num_predicates), requires_grad=True)
#embedding1= torch.Tensor([1,0]).view(1,-1)
#embeddings2= torch.Tensor([0,1]).expand(3,2)
#embeddings= Variable(torch.cat((embedding1,embeddings2),0) ,requires_grad=True)
embeddings = Variable(torch.rand(num_predicates, num_feat), requires_grad=True)

#rule1 = torch.Tensor([1,0,1,0,1,0]).view(1,-1)
#rule2 = torch.Tensor([0,1,1,0,0,1]).view(1,-1)
#rules = Variable(torch.cat((rule1,rule2),0), requires_grad=True)
rules = Variable(torch.rand(num_rules, num_feat*3), requires_grad=True)

#encoder_m = Encoder()
#core_init = 'from_data'
core_init = 'random'


#initial_val = Variable(data.clone(), requires_grad=True)
if core_init == 'from_data':
    initial_val = [Variable(dataIS_A.clone(), requires_grad=True),                    Variable(data_is.clone(), requires_grad=True),                    Variable(data_can.clone(), requires_grad=True),                    Variable(data_has.clone(), requires_grad=True)]
elif core_init == 'random':
    initial_val = [Variable(torch.rand(dataIS_A.size()), requires_grad=True),                    Variable(torch.rand(data_is.size()), requires_grad=True),                    Variable(torch.rand(data_can.size()), requires_grad=True),                    Variable(torch.rand(data_has.size()), requires_grad=True)]


optimizer = torch.optim.Adam([
        {'params': initial_val},
        {'params': [rules], 'lr': learning_rate_rules},
        {'params': [embeddings], 'lr':learning_rate_rules},
    ], lr = learning_rate)

criterion = torch.nn.BCELoss(size_average=False)


targetIS_A = Variable(dataIS_A)
target_is = Variable(data_is)
target_can = Variable(data_can)
target_has = Variable(data_has)
target = [targetIS_A, target_is, target_can, target_has]

beta = 10./total_elements
lamb= 1./total_elements  #1.5 works

print('target', target)

for epoch in range(num_iters):
    for par in optimizer.param_groups[:]:
        for param in par['params']:
            param.data.clamp_(min=0.,max=1.)
            
    optimizer.zero_grad()
    
    valuation = initial_val
    
    loss_reg = Variable(torch.Tensor([0]))
    for predicate in intensional_predicates:
        #loss_reg += torch.sum(torch.ge(valuation,0.4).type(torch.FloatTensor))
        #loss_reg += torch.sum(valuation)
        loss_reg += torch.sum(valuation[predicate]/(valuation[predicate]+2.))
    
    #if epoch % 49 == 0 and epoch>0 :
    #    print('epoch {} before_decoder'.format(epoch), torch.round(100*valuation)/100.)
        
    for step in range(steps):
        valuation = decoder_efficient(valuation)
        #if epoch % 49 == 0 and epoch>0 :
        #    print('epoch {}, step {}'.format(epoch,step+1), torch.round(100*valuation)/100.)
    
    loss = Variable(torch.Tensor([0]))
    for predicate in intensional_predicates:
        loss += criterion(valuation[predicate],target[predicate]) 
    loss_pos = loss.clone()
    loss = beta*loss
    loss += lamb*loss_reg 
    
    
    print(epoch, 'losssssssssssssssssssss',torch.round(10000*loss).data[0]/10000,                 'loss_pos', torch.round(10000*loss_pos).data[0]/10000,                  'loss_reg', torch.round(10000*loss_reg).data[0]/10000)
    loss.backward()
    optimizer.step()

#data[0,6,0],data[1,6,0],data[1,6,2],data[1,6,6] = 0,0,0,0
# Knows: salmon is fish, salmon is salmon

print(valuation)
print(embeddings)
print(rules)



# In[43]:

pdb.set_trace()
print(embeddings)
print(rules)

rules_aux = torch.cat((rules[:,:num_feat],rules[:,num_feat:2*num_feat],rules[:,2*num_feat:3*num_feat]),0)
rules_aux = rules_aux.repeat(num_predicates,1)
embeddings_aux = embeddings.repeat(1,num_rules*3).view(-1,num_feat)
#print('revisin',rules_aux,embeddings_aux,F.cosine_similarity(embeddings_aux, rules_aux))


unifs_real = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)
#####Unifs real is of the form
#####predicates * [Head1 Head2 Body1 Body21 Body2 Body22], so i normalize in dim,0  

print('aaaaa',unifs_real)
#print('oooo',F.pairwise_distance(embeddings_aux, rules_aux).view(num_predicates,-1))

unifs = F.pairwise_distance(embeddings_aux, rules_aux).view(num_predicates,-1)
unifs = torch.exp(-unifs)
#print(unifs)
unifs_sum = torch.sum(unifs, 0)
unifs= unifs/unifs_sum
print('finaaaal',unifs)

print('normalize',F.normalize(unifs_real,dim=0))



# In[20]:

rules[0][0]

