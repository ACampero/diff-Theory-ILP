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

##Background Knowledge
zero_extension = torch.zeros(1,num_constants)
zero_extension[0,0] = 1
succ_extension = torch.eye(num_constants-1,num_constants-1)
succ_extension = torch.cat((torch.zeros(num_constants-1,1),succ_extension),1)
succ_extension = torch.cat((succ_extension,torch.zeros(1,num_constants)),0)

#Intensional Predicates
aux_extension = torch.zeros(num_constants,num_constants)
even_extension = torch.zeros(1,num_constants)

##Valuation
valuation_init = [Variable(zero_extension), Variable(succ_extension), Variable(aux_extension), Variable(even_extension)]

num_predicates= len(valuation_init)
intensional_predicates=[2,3]
num_intensional_predicates = len(intensional_predicates)

##Target
target = Variable(torch.zeros(1,num_constants))
even = [0,2,4,6]
for integer in even:
    target[0,integer]=1


##------FORWARD CHAINING------
def decoder_efficient(valuation, step):
    ##Unifications
    rules_aux = torch.cat((rules[:,:num_feat],rules[:,num_feat:2*num_feat],rules[:,2*num_feat:3*num_feat]),0)
    rules_aux = rules_aux.repeat(num_predicates,1)
    embeddings_aux = embeddings.repeat(1,num_rules*3).view(-1,num_feat)    
    unifs = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)
    
    ##Get_Valuations
    valuation_new = [valuation[i] ]
    valuation_new = [deepcopy(valuation[0]),deepcopy(valuation[1]),Variable(torch.zeros(valuation[2].size())),Variable(torch.zeros(valuation[3].size()))]
    for predicate in intensional_predicates:
        if valuation[predicate].size()[0] == 1:
            for s in range(num_constants):
                valuation_aux = Variable(torch.Tensor([0]))
                for body1 in range(num_predicates):
                    for body2 in range(num_predicates):
                        ## Get nums
                        if valuation[body1].size()[0] == 1:
                            if valuation[body2].size()[0] == 1:
                                num = torch.min(valuation[body1][0,s],valuation[body2][0,s])
                            else: 
                                num = torch.min(valuation[body1][0,:],valuation[body2][:,s])
                                num = torch.max(num)
                        else:
                            if valuation[body2].size()[0] == 1:
                                num = torch.min(valuation[body1][:,s],valuation[body2][0,s])
                                num = torch.max(num)
                            else:
                                num = 0


                        ## max across three rules
                        new = Variable(torch.Tensor([0]))
                        for rule in range(num_rules): 
                            unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                            new = torch.max(new,unif) 

                        num = num*new 
                        valuation_aux = torch.max(valuation_aux, num)
                valuation_new[predicate][0,s] = torch.max(valuation[predicate][0,s], valuation_aux) 
            
            
            
        else:
            for s in range(num_constants):
                for o in range(num_constants):
                    valuation_aux = Variable(torch.Tensor([0]))
                    for body1 in range(num_predicates):
                        for body2 in range(num_predicates):
                            ## Get nums
                            if valuation[body1].size()[0] == 1:
                                if valuation[body2].size()[0] == 1:
                                    num = torch.min(valuation[body1][0,s],valuation[body2][0,o])
                                else: 
                                    num = torch.min(valuation[body1][0,s],valuation[body2][s,o])
                                    #num = torch.max(num)
                            else:
                                if valuation[body2].size()[0] == 1:
                                    num = torch.min(valuation[body1][s,o],valuation[body2][0,o])
                                    #num = torch.max(num)
                                else: 
                                    num = torch.min(valuation[body1][s,:],valuation[body2][:,o])
                                    num = torch.max(num)

                            ## max across three rules
                            new = Variable(torch.Tensor([0]))
                            for rule in range(num_rules): 
                                unif = unifs[predicate][rule]*unifs[body1][num_rules+rule]*unifs[body2][2*num_rules+rule]
                                new = torch.max(new,unif)
                                #could be amalgamate

                            num = num*new 
                            valuation_aux = torch.max(valuation_aux, num)
                    valuation_new[predicate][s,o] = torch.max(valuation[predicate][s,o], valuation_aux) 
                
    return valuation_new
            
def amalgamate(x,y):
    return x + y - x*y

##------SETUP------
num_iters = 100
learning_rate = .1
steps = 4
num_feat=4
num_rules = 3
num_predicates= 4
intensional_predicates=[2,3]
num_intensional_predicates = len(intensional_predicates)

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
    optimizer.zero_grad()
    valuation = valuation_init

    for step in range(steps):
        valuation = decoder_efficient(valuation,step)
        #print('step',step,'valuation3', valuation[3], 'valuation2',valuation[2])

    loss = criterion(valuation[-1][0,:],target[0,:])
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

print(target[0:])
print('val',valuation[-1])
accu = torch.sum(torch.abs(torch.round(valuation[-1][0,:])-target[0,:])).data[0]
print('accuracy',accu)

pdb.set_trace()
