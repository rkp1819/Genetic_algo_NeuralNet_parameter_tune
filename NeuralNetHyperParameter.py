#Tuning hyper parameters of neural network using GA
import random
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import pickle

MAX_ITR = 50

class OptimalNeuralNet(object):
    ELITE = None
    pop_size = None
    model = None
    generation = None
    activation_functions = ['relu', 'selu', 'sigmoid', 'tanh', 'elu', 'softplus', 'softsign',
                            'hard_sigmoid', 'linear','softmax']
    optimizer = ['SGD']
    """, 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']"""
    regressor_loss = ['mean_squared_error']
    """, 'mean_absolute_error', 'mean_absolute_percentage_error',
                      'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'logcosh']"""
    classifier_loss = ['categorical_hinge', 'categorical_crossentropy', 'sparse_categorical_crossentropy',
                       'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']
    metrics = ['mae']
    """, 'sparse_categorical_accuracy']"""

    def classification_accuracy_percentage(test_y, pred):
        N = len(test_y) 
        count = 0
        for i in range(N):
            if(pred[i] == test_y[i]):
                count+=1
            
        return count/N*100


    def __init__(self, train_x, train_y, test_x, test_y, pop_size = 10, model = 'regressor'):
        self.pop_size = pop_size
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.ELITE = {'cost':(100, 1)}
        
    def initialize(self,):
        self.generation = []
        for i in range(self.pop_size):
           self.generation.append( {'input':(10, random.choice(self.activation_functions)),
                                    'output':(1, random.choice(self.activation_functions)),
                                    'layers':[(random.randint(1, 10), random.choice(self.activation_functions))
                                         for layer in range(random.randint(1, 3))],
             'optimizer':random.choice(self.optimizer),
             'loss':random.choice(self.regressor_loss if self.model is 'regressor' else self.classifier_loss),
             'metrics':self.metrics[0] if self.model is 'regressor' else self.metrics[1],
             'cost': (100, 1)
             })
        return self.generation
        
    def selection(self,):
        print('in selection')
        self.generation = self.evaluate_costs()
        new_generation = []
        for i in range(self.pop_size):
            u = self.generation[random.randint(0, self.pop_size-1)]
            v = self.generation[random.randint(0, self.pop_size-1)]
            if(round(u['cost'][0], 1) >= round(v['cost'][0], 1) and round(u['cost'][1], 2) > round(v['cost'][1]), 2):
                new_generation.append(v)
                if round(v['cost'][0], 1) <= round(self.ELITE['cost'][0], 1) and round(v['cost'][1], 2) < round(self.ELITE['cost'][1], 2):
                    self.ELITE = v
            else:
                new_generation.append(u)
                if round(u['cost'][0], 1) <= round(self.ELITE['cost'][0], 1) and round(u['cost'][1], 2) < round(self.ELITE['cost'][1], 2):
                    self.ELITE = u
        self.generation = new_generation
        return self.generation

    def evaluate_costs(self,):
        print("evaluating_costs")
        for i in range(self.pop_size):
            blue_print = self.generation[i]
##            print(blue_print)
            model = Sequential()
            model.add(Dense(blue_print['input'][0], input_dim = 10, activation = blue_print['input'][1]))
            for layer in blue_print['layers']:
                model.add(Dense(layer[0], activation = layer[1]))
            model.add(Dense(blue_print['output'][0],activation = blue_print['output'][1]))
            model.compile(loss = blue_print['loss'], optimizer = blue_print['optimizer'], metrics = [blue_print['metrics']])
            model.fit(self.train_x, self.train_y, epochs = 2, batch_size = 2, verbose = 0)
            scores = model.evaluate(self.test_x, self.test_y, batch_size = 2, verbose = 0)
            print(f"{model.metrics_names[0]}, {scores[0]} , {model.metrics_names[1]}, {scores[1]}")
            self.generation[i]['cost'] = (scores[0],scores[1])
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
            u_copy['metrics'] = [u['metrics'],v['metrics']]
            merged = u_copy
            u = {'input': random.choice(merged['input']),
                 'output': random.choice(merged['output']),
                 'layers': random.choices(merged['layers'],
                    k = random.randint(1,len(merged['layers']) if len(merged['layers'])<=3 else 3)),
                 'optimizer': random.choice(merged['optimizer']),
                 'loss': random.choice(merged['loss']),
                 'metrics': random.choice(merged['metrics']),
                 'cost': (100, 1)}
            v = {'input': random.choice(merged['input']),
                 'output': random.choice(merged['output']),
                 'layers': random.choices(merged['layers'],
                    k = random.randint(1,len(merged['layers']) if len(merged['layers'])<=3 else 3)),
                 'optimizer': random.choice(merged['optimizer']),
                 'loss': random.choice(merged['loss']),
                 'metrics': random.choice(merged['metrics']),
                 'cost': (100, 1)}
            new_generation.append(u)
            new_generation.append(v)
        self.generation = new_generation
        return self.generation

    def mutation(self,):
        self.generation[random.randint(1, self.pop_size-1)] = {'input':(10, random.choice(self.activation_functions)),
                                                               'output':(1, random.choice(self.activation_functions)),
             'layers': [(random.randint(1, 10), random.choice(self.activation_functions))
                                         for layer in range(random.randint(1, 3))],
             'optimizer': random.choice(self.optimizer),
             'loss': random.choice(self.regressor_loss if self.model is 'regressor' else self.classifier_loss),
             'metrics': self.metrics[0] if self.model is 'regressor' else self.metrics[1],
             'cost': (100, 1)
             }
        return self.generation
    
    def terminate(self,):
        terminate = True
        cost = self.generation[0]['cost']
        if(cost is 1):
           return not terminate
        for i in range(self.pop_size):
            if round(self.generation[i]['cost'][0], 1) != round(cost[0], 1) or round(self.generation[i]['cost'][1], 2) != round(cost[1], 2):
                terminate = False
                break
        return terminate
    
    def optimized_solution(self,):
        cost = self.generation[0]['cost']
        index = 0
        for i in range(self.pop_size):
            if round(self.generation[i]['cost'][0], 1) <= round(cost[0], 1) and round(self.generation[i]['cost'][1], 2) < round(cost[1], 2):
                cost = self.generation[i]['cost']
                index = i
        return index
    
if __name__ == '__main__':
    data_set = pd.DataFrame(pd.read_csv('D:/Intrests/Project 14067/dataset/hrl_load_metered_ME.csv'))
    weather_data = pd.DataFrame(pd.read_csv('D:/Intrests/Project 14067/dataset/date_time_weather_ME_std.csv'))
    data_set = data_set[['mw']]
    X_train, X_test = weather_data[65000:70029], weather_data[70029:]
    Y_train, Y_test = data_set[65000:70029], data_set[70029:]
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    nn = OptimalNeuralNet(X_train, Y_train, X_test, Y_test, pop_size = 10)   
    nn.generation = nn.initialize()
    i=0
    while(not nn.terminate() and i<MAX_ITR):
        print("iter", i)
        nn.generation = nn.selection()
        nn.generation = nn.crossover()
        nn.generation = nn.mutation()
        i+=1
    nn.generation = nn.selection()
    print(f'The optimized solution is : {nn.generation[nn.optimized_solution()]}, ELITE : {nn.ELITE}')

    save_nn = open("OptimalNN.pickle", 'wb')
    pickle.dump(nn, save_nn)
    save_nn.close()














