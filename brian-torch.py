#%%
import torch
import numpy as np
import random
import csv
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt

def writeCSV(filename,  fieldnames, dataset):
    with open(filename + '.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        count = 0
        for value in dataset:
            if count%10000 == 0:
                print(filename + ': wrote ' + str(count) + ' datapoints')
            
            row = {}
            fieldcount = 0
            for field in fieldnames:
                row[field] = str(value[count])
                fieldcount += 1

            writer.writerow(row)
            
        count += 1

np.random.seed(12)
num_observations = 50000

x_data = []
x_fieldnames = []
y_data = []
y_fieldname = ['y']
for x in range (2000):
    x_fieldnames.append('x' + str(x))
    set1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x_data.append(set1.flatten())
    y_data.append(0)
    set2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    x_data.append(set2.flatten())
    y_data.append(1)
    if x % 1000 == 0:
        print(x)
print("finished generating")

writeCSV("x_data", x_fieldnames, x_data)

#Tutorial
device = torch.device("cuda:0")

class LogisticRegression(torch.nn.Module):    
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_observations*2, 1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
        
model = LogisticRegression()
model = torch.nn.DataParallel(model)
model.to(device)

x_tensor = Variable(torch.Tensor(x_data)).to(device)
y_tensor = Variable(torch.Tensor(y_data)).to(device)

print("running torch")
#%%
#model = LogisticRegression()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
   model.train()
   optimizer.zero_grad()
   y_pred = model(x_tensor)
   loss = criterion(y_pred, y_tensor.unsqueeze(1))
   loss.backward()
   optimizer.step()

new_x = Variable(torch.Tensor([[4.0]]))
y_pred = model(new_x)
print("predicted Y value: ", y_pred.data[0][0])