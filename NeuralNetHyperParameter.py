#Tuning hyper parameters of neural network using GA
import random
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

MAX_ITR = 10

class OptimalNeuralNet(object):
    ELITE = None
    pop_size = None
    model = None
    generation = None
    activation_functions = ['relu', 'selu', 'sigmoid', 'tanh', 'elu', 'softplus', 'softsign',
                            'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ThresholdedReLU','Softmax']
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    regressor_loss = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                      'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'logcosh']
    classifier_loss = ['categorical_hinge', 'categorical_crossentropy', 'sparse_categorical_crossentropy',
                       'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
    metrices = ['mae', 'sparse_categorical_accuracy']

    def classification_accuracy_percentage(test_y, pred):
        N = len(test_y) 
        count = 0
        for i in range(N):
            if(pred[i] == test_y[i]):
                count+=1
            
        return count/N*100


    def __init__(self, pop_size = 10, model = 'regressor', train_x=None, train_y=None, test_x=None, test_y=None):
        self.pop_size = pop_size
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.ELITE = {'cost':1}
        
    def initialize(self,):
        self.generation = []
        for i in range(self.pop_size):
           self.generation.append( {'layers':[(random.randint(1, 10), random.choice(self.activation_functions))
                                         for layer in range(random.randint(1, 3))],
             'optimizer':random.choice(self.optimizer),
             'loss':random.choice(self.regressor_loss if self.model is 'regressor' else self.classifier_loss),
             'metrices':self.metrices[0] if self.model is 'regressor' else self.metrices[1],
             'cost': 1
             })
        return self.generation
        
    def selection(self,):
        self.generation = self.evaluate_costs()
        new_generation = []
        for i in range(self.pop_size):
            u = self.generation[random.randint(0, self.pop_size-1)]
            v = self.generation[random.randint(0, self.pop_size-1)]
            if(u['cost']>v['cost']):
                new_generation.append(v)
                if v['cost']<self.ELITE['cost']:
                    self.ELITE = v
            else:
                new_generation.append(u)
                if u['cost']<self.ELITE['cost']:
                    self.ELITE = u
        self.generation = new_generation
        return self.generation

    def evaluate_costs(self,):
        for i in range(self.pop_size):
            self.generation[i]['cost'] = np.random.rand()
        return self.generation

    
    def crossover(self,):
        new_generation = []
        for i in range(self.pop_size//2):            
            u = self.generation[random.randint(0, self.pop_size-1)]
            v = self.generation[random.randint(0, self.pop_size-1)]
            u_copy = u.copy()
            v_copy = v.copy()
            u_copy['layers'].extend(v['layers'])
            u_copy['optimizer'] = [u['optimizer'],v['optimizer']]
            u_copy['loss'] = [u['loss'], v['loss']]
            u_copy['metrices'] = [u['metrices'],v['metrices']]
            merged = u_copy
            u = {'layers': random.choices(merged['layers'],
                    k = random.randint(1,len(merged['layers']) if len(merged['layers'])<=3 else 3)),
                 'optimizer': random.choice(merged['optimizer']),
                 'loss': random.choice(merged['loss']),
                 'metrices': random.choice(merged['metrices']),
                 'cost': 1}
            v = {'layers': random.choices(merged['layers'],
                    k = random.randint(1,len(merged['layers']) if len(merged['layers'])<=3 else 3)),
                 'optimizer': random.choice(merged['optimizer']),
                 'loss': random.choice(merged['loss']),
                 'metrices': random.choice(merged['metrices']),
                 'cost': 1}
            new_generation.append(u)
            new_generation.append(v)
        self.generation = new_generation
        return self.generation

    def mutation(self,):
        self.generation[random.randint(1, self.pop_size-1)] = {'layers': [(random.randint(1, 10),
                                                                          random.choice(self.activation_functions))
                                         for layer in range(random.randint(1, 3))],
             'optimizer': random.choice(self.optimizer),
             'loss': random.choice(self.regressor_loss if self.model is 'regressor' else self.classifier_loss),
             'metrices': self.metrices[0] if self.model is 'regressor' else self.metrices[1],
             'cost': 1
             }
        return self.generation
    
    def terminate(self,):
        terminate = True
        cost = self.generation[0]['cost']
        if(cost is 1):
           return not terminate
        for i in range(self.pop_size):
            if round(self.generation[i]['cost'],2) != round(cost,2):
                terminate = False
                break
        return terminate
    
    def optimized_solution(self,):
        cost = self.generation[0]['cost']
        index = 0
        for i in range(self.pop_size):
            if(round(self.generation[i]['cost'], 2) < round(cost, 2)):
                cost = self.generation[i]['cost']
                index = i
        return index
    
if __name__ == '__main__':
    
    
    nn = OptimalNeuralNet()    
    nn.generation = nn.initialize()
    i=0
    while(not nn.terminate() and i<MAX_ITR):
        nn.generation = nn.selection()
        nn.generation = nn.crossover()
        nn.generation = nn.mutation()
        i+=1
    nn.generation = nn.selection()
    print(f'The optimized solution is : {nn.generation[nn.optimized_solution()]}, ELITE : {nn.ELITE}')
















