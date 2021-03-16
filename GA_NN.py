import random
import pickle
import numpy as np


def create_pop(feature_len, pop_size=30):
    return [[0 if random.random() > 0.5 else 1 for _ in range(feature_len)] for _ in range(pop_size)]


def create_pop_net(max_num_layers, min_num_neurons, max_num_neurons, pop_size=30):
    l = list(range(min_num_neurons, max_num_neurons))
    return [np.random.choice(l, random.randint(2, max_num_layers)) for _ in range(pop_size)]


def create_pop_activations(pop_net, num_activations):
    pop_activations = []
    l = list(range(0, num_activations))
    for p in pop_net:
        pop_activations.append(np.random.choice(l, len(p)))
    return pop_activations


def recombinant(p1, p2, fit1, fit2):
    threshold = fit1 / (fit1 + fit2)
    return [p1[i] if random.random() > threshold else p2[i] for i in range(len(p1))]


def recombinant_net(p1, p2, activations1, activations2, fit1, fit2):
    fit1, fit2 = fit1**2, fit2**2
    threshold = fit1 / (fit1 + fit2)
    rest = []
    rest_ac = []
    min_len = min(len(p1), len(p2))
    if fit1 > fit2 and len(p1) > len(p2):
        rest = p1[len(p2):]
        rest_ac = activations1[len(p2):]
    if fit1 < fit2 and len(p1) < len(p2):
        rest = p2[len(p1):]
        rest_ac = activations2[len(p1):]
    net = [p1[i] if random.random() > threshold else p2[i] for i in range(min_len)] + rest
    activation = [activations1[i] if random.random() > threshold else activations2[i] for i in
                  range(min_len)] + rest_ac
    return net, activation


def mutate(p1):
    threshold = 0.1
    return [item if random.random() > threshold else (1 - item) for item in p1]


def mutate_net(p, activations):
    threshold = 0.1
    res = []
    for item in p:
        rand = random.random()
        to_add1 = 1 if rand < threshold else 0
        to_add2 = -1 if random.random() > (1-threshold) else 0
        new_val = item + to_add1 + to_add2
        if new_val > 0:
            res.append(new_val)
    if random.random() < threshold:
        res.append(random.randint(min(res), max(res)))
        activations.append(random.randint(0, 2))
    if len(res) == 0:
        res = [1]
        activations = [1]
    return res, activations


def calc_fitness(p, p_net, p_activations, model_type, train_df, test_df, train_y, test_y, all_cols_for_model, length_penalty,
                 val_df, val_y, activation_funcs):
    cols_for_model = [all_cols_for_model[i] for i in range(len(all_cols_for_model)) if p[i] == 1]
    train_x = train_df[cols_for_model]
    val_x = val_df[cols_for_model]
    test_x = test_df[cols_for_model]
    model = model_type(train_x, val_x, test_x, train_y, val_y, test_y)
    model.create_model(p_net, p_activations,  activation_funcs)
    model.train()
    predictions = model.predict_test()
    num_of_ones = sum(p)
    return (1 / model.calc_loss(predictions)) - length_penalty*num_of_ones


def find_next(cum_sum_list):
    threshold = random.random()
    i = 0
    while threshold < cum_sum_list[i]:
        i += 1
    return i


def to_loss(fit, member, length_penalty):
    return 1/(fit + sum(member)*length_penalty)


def print_best_loss(i, best_loss, best_member, best_member_net, best_activations, funcs):
    print("i = {0}, best loss = {1}".format(i, best_loss))  # because we wnt to see the actual loss
    print("best net = {0}".format(best_member_net))
    print("best activations = {0}".format([funcs[k] for k in best_activations]))
    print("num of features = {0}".format(sum(best_member)))
    print()


def iterate(model, num_iter, feature_len, pop_size, train_df, test_df, texture_cols, cols_for_model, length_penalty,
            max_num_layers, min_num_neurons, max_num_neurons, val_df, activation_funcs):
    pop = create_pop(feature_len, pop_size)
    pop_net = create_pop_net(max_num_layers, min_num_neurons, max_num_neurons, pop_size)
    pop_activations = create_pop_activations(pop_net, 3)
    # pop, pop_net, pop_activations = load_pop()
    leave_val = int(pop_size * 0.2)
    train_y, val_y, test_y = train_df[texture_cols], val_df[texture_cols], test_df[texture_cols]
    best_fit = 0.0000001
    best_member = [0]*feature_len
    best_member_net = [1, 1, 1]
    best_member_activation = [1, 1, 1]
    for i in range(num_iter):
        fitness = [calc_fitness(pop[i], pop_net[i], pop_activations[i], model, train_df, test_df, train_y, test_y, cols_for_model,
                                length_penalty, val_df, val_y, activation_funcs) for i in range(len(pop))]
        # print(fitness)
        # print(pop)
        # print([sum(p) for p in pop])
        # print(pop_net)
        # print(pop_activations)
        # print(to_loss(best_fit, best_member, length_penalty))
        # print(to_loss(max(fitness), pop[fitness.index(max(fitness))], length_penalty))
        # print()
        if to_loss(best_fit, best_member, length_penalty) > to_loss(max(fitness), pop[fitness.index(max(fitness))],
                                                                    length_penalty):
            best_fit = max(fitness)
            best_member = pop[fitness.index(best_fit)]
            best_member_net = pop_net[fitness.index(best_fit)]
            best_member_activation = pop_activations[fitness.index(best_fit)]
        if i % 5 == 0:
            print_best_loss(i, to_loss(best_fit, best_member, length_penalty), best_member, best_member_net,
                            best_member_activation, activation_funcs)
        fitness = [f**3 for f in fitness]
        total = sum(fitness)
        cum_sum = [sum(fitness[:i]) / total for i in range(len(fitness))]
        new_pop = [p for _, p in sorted(zip(fitness, pop), reverse=True)][:leave_val]
        new_pop_net = [p for _, p in sorted(zip(fitness, pop_net), reverse=True)][:leave_val]
        new_pop_activations = [p for _, p in sorted(zip(fitness, pop_activations), reverse=True)][:leave_val]
        for _ in range(len(pop) - leave_val):
            index1, index2 = find_next(cum_sum), find_next(cum_sum)
            rec = recombinant(pop[index1], pop[index2], fitness[index1], fitness[index2])
            new_pop.append(mutate(rec))
            rec_net, rec_activation = recombinant_net(pop_net[index1], pop_net[index2], pop_activations[index1],
                                                      pop_activations[index2], fitness[index1], fitness[index2])
            mu_net, mu_activation = mutate_net(rec_net, rec_activation)
            new_pop_net.append(mu_net)
            new_pop_activations.append(mu_activation)
        pop = new_pop
        pop_net = new_pop_net
        pop_activations = new_pop_activations
        save_pop(pop, pop_net, pop_activations)
    return best_member, to_loss(best_fit, best_member, length_penalty), best_member_net, best_member_activation


def load_pop():
    with open('pop', 'rb') as f:
        pop = pickle.load(f)
    with open('pop_net', 'rb') as f:
        pop_net = pickle.load(f)
    with open('pop_activations', 'rb') as f:
        pop_activations = pickle.load(f)
    return pop, pop_net, pop_activations


def save_pop(pop, pop_net, pop_activations):
    with open('pop', 'wb') as f:
        pickle.dump(pop, f)
    with open('pop_net', 'wb') as f:
        pickle.dump(pop_net, f)
    with open('pop_activations', 'wb') as f:
        pickle.dump(pop_activations, f)

