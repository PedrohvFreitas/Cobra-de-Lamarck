import numpy as np
import torch
import random
import math

class EvolutiveAlgorithm():
    def __init__(self, k):
        self.k = k

    def selection_mechanism(self, agents_dict):
        """Seleciona os melhores 10 melhores com base no score que ela fizeram (Elitismo)"""
        pairs = []
        agents_dict = sorted(agents_dict.items(), key= lambda item: item[1], reverse=True)
        agents_dict_key = [item[0] for item in agents_dict]
        
        agents_dict_key = agents_dict_key[0: self.k]
        agents_dict_key_2 = agents_dict_key[self.k: 2 * self.k]
        
        agents_dict_key_shuffled = sorted(agents_dict_key, key= lambda x: random.random())

        for j in range(0, len(agents_dict_key), 2):
            pairs.append([agents_dict_key_shuffled[j], agents_dict_key_shuffled[j+1]])        
        return agents_dict_key, pairs, agents_dict_key_2

    def reproduction(self, snakes):
        """Realiza o crossover dos casais selecionados via média ponderada de seus pesos e bias,
           utilizando o score como fator de ponderação"""
        baby_snake_dict = {}

        alpha = snakes[0].score / (snakes[0].score + snakes[1].score + 1e-12)
        beta = snakes[1].score / (snakes[0].score + snakes[1].score + 1e-12)

        params_0 = snakes[0].model.state_dict()
        params_1 = snakes[1].model.state_dict()
        keys = params_0.keys()
            
        for key in keys:
            baby_snake_dict[key] = alpha * params_0[key] + beta * params_1[key]
        
        return baby_snake_dict

    def mutation(self, gaussian, snake, mutation_percentage):
        """Introduz variabilidade genética na rede neural através de uma distribuição normal. 
        Utiliza uma estratégia de mutação dual: 'Big Mutation' para diversificação de agentes estagnados 
        e 'Small Mutation' para preservação e otimização dos melhores"""
        param = snake.model.state_dict()
        keys = param.keys()

        for key in keys:
            
            quantity_weights = int(param[key].numel() * mutation_percentage)
            random_vector = np.random.randint(0, param[key].numel(), size=quantity_weights)
            if gaussian ==  'big':
                gaussian_vector = np.random.normal(0, 0.1, 2000)
            else:
                gaussian_vector = np.random.normal(0, 0.01, 2000)

            for k in range(len(random_vector)):
                if param[key].ndim == 1:
                    index = random_vector[k]
                    param[key][index] += gaussian_vector[k]
                else:
                    collum = math.floor(random_vector[k] / len(param[key]))
                    line = random_vector[k] % len(param[key])

                    param[key][line, collum] += gaussian_vector[k]
                
            snake.model.load_state_dict(param)

            return snake