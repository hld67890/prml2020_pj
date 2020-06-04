import numpy as np
import torch
import torch.nn as nn

class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1024, 2)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax ( dim = 1 )

    def forward(self, x):
        '''
        Please finish your code here.
        '''
        #print ( x.shape )
        x = self.dense ( x )
        x = self.sig ( x )
        x = self.soft ( x )
        return x


def compute_loss(logits, labels):
    #print ( logits[0] , logits[1] )
    #print ( logits.view(-1,2) , labels.view(-1) )
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1,2), labels.view(-1))


def train_one_step(model, optimizer, x, label):
    model.train()
    optimizer.zero_grad()
    #print ( torch.tensor(x).shape )
    logits = model(torch.tensor(x,dtype=torch.float32))
    loss = compute_loss(logits, torch.tensor(label,dtype=torch.long))

    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


def train(steps, model, optimizer,data):
    #print ( data )
    loss = 0.0
    for step in range(steps):
        x,label = data
        loss = train_one_step(model, optimizer, x, label)
        if step % 200 == 0:
            print('step', step, ': loss', loss)

    return loss


def evaluate(model,data):
    x,label = data
    with torch.no_grad():
        logits = model(torch.tensor(x,dtype=torch.float32))
    logits = logits.numpy()
    pred = logits

    #cheat
    for i in range(pred.shape[0]):
        if pred[i][1] > 0.5: pred[i][1] = 1
        else: pred[i][1] = 0

    #for o in list(zip(datas[2], res))[:20]:
    #    print(o[0],"@@", o[1], o[0]==o[1])

    #print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))
    return pred

def trainmodel( data ):
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer , data )
    return model
