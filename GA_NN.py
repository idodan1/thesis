import random
from functions import *
import os


def create_pop(feature_space, permutation_function, pop_size=30):
    return [permutation_function(feature_space) for _ in range(pop_size)]


def recombinant(p1, p2, fit1, fit2):
    threshold = fit1 / (fit1 + fit2)
    son = {}
    list_winner = 0 if random.random() > threshold else 1
    for key in p1:
        if type(p1[key]) == int:
            son[key] = p1[key] if random.random() > threshold else p2[key]
        elif type(p1[key]) == str:
            son[key] = p1[key]
        else:
            son[key] = p1[key] if list_winner == 0 else p2[key]
    return son


def mutate(p1, feature_space, threshold):
    new_son = {}
    for key in p1:
        if type(p1[key]) == int:
            new_son[key] = p1[key] if random.random() > threshold else max(1, p1[key]+np.random.randint(-2, 2))
            if new_son[key] < feature_space[key][0]:
                new_son[key] = feature_space[key][0]
            if new_son[key] > feature_space[key][1]-1:
                new_son[key] = feature_space[key][1]-1
        elif type(p1[key]) == str:
            new_son[key] = p1[key]
        else:
            new_son[key] = [val if random.random() > threshold else
                            np.random.randint(feature_space[key][0], feature_space[key][1]) for val in p1[key]]
    return new_son


def calc_fitness(p, train_x, test_x, train_y, test_y, model_type, test_df, texture_cols):
    model = model_type(train_x, test_x, train_y, test_y, p)
    model.create_model()
    model.train()
    predictions = model.predict_test()
    res_model = calc_rmses(predictions, test_df, texture_cols)
    return flip_fitness(res_model.sum())


def flip_fitness(val):
    return 1/val


def find_next(cum_sum_list):
    threshold = random.random()
    i = 0
    while threshold < cum_sum_list[i]:
        i += 1
    return i


def loss_stats(i, best_fit, best_currently, worst_fit, model_name, texture_type):
    stats_file = 'results_all_models/{0}/loss_stats-{1}.xlsx'.format(model_name, texture_type)
    if os.path.isfile(stats_file):
        stats_df = pd.read_excel(stats_file)
    else:
        stats_df = pd.DataFrame(columns=['generation', 'best', 'worst'])
    row = pd.DataFrame.from_dict({'generation': [i], 'best': [best_fit], 'worst': [worst_fit]})
    stats_df = pd.concat([stats_df, row], sort=True)
    stats_df.to_excel(stats_file, index=False)
    print("i = {0}, best rmse = {1}".format(i, best_fit))
    print('best currently = {0}'.format(best_currently))
    print()


def save_top_5(pop, fitness, model_name, texture_type):
    file_name = 'results_all_models/{0}/top5-{1}.xlsx'.format(model_name, texture_type)
    cols = ['fitness', 'rmse', 'features']
    if os.path.isfile(file_name):
        res_df = pd.read_excel(file_name)
    else:
        res_df = pd.DataFrame(columns=cols)
    for i in range(min(5, len(fitness))):
        res_dict = {'fitness': [fitness[i]], 'rmse': [flip_fitness(fitness[i])],
                    'features': [str(pop[i])]}
        row = pd.DataFrame.from_dict(res_dict)
        res_df = pd.concat([res_df, row], sort=True)
    res_df = res_df.sort_values(by=['rmse'])
    res_df = res_df[0:5]
    res_df.to_excel(file_name, index=False)


def save_worst_5(pop, fitness, model_name, texture_type):
    file_name = 'results_all_models/{0}/worst5-{1}.xlsx'.format(model_name, texture_type)
    cols = ['fitness', 'rmse', 'features']
    if os.path.isfile(file_name):
        res_df = pd.read_excel(file_name)
    else:
        res_df = pd.DataFrame(columns=cols)
    for i in range(1, min(5+1, len(fitness)+1)):
        res_dict = {'fitness': [fitness[-i]], 'rmse': [flip_fitness(fitness[-i])],
                    'features': [str(pop[-i])]}
        row = pd.DataFrame.from_dict(res_dict)
        res_df = pd.concat([res_df, row], sort=True)
    res_df = res_df.sort_values(by=['fitness'])
    res_df = res_df[0:5]
    res_df.to_excel(file_name, index=False)


def iterate(model_type, feature_space, train_x, test_x, train_y, test_y, test_df, texture_cols,
            pop_size, permutation_function, num_iter, model_name, leave_val, texture_type,
            threshold_mutation):
    pop = create_pop(feature_space, permutation_function, pop_size)
    best_fit = 100
    best_member = {}
    for i in range(num_iter):
        fitness = [calc_fitness(p, train_x, test_x, train_y, test_y, model_type, test_df, texture_cols)
                   for p in pop]
        fitness, pop = (list(t) for t in zip(*sorted(zip(fitness, pop), reverse=True)))
        save_top_5(pop, fitness, model_name, texture_type)
        save_worst_5(pop, fitness, model_name, texture_type)
        best_fit_iter = flip_fitness((max(fitness)))
        worst_fit_iter = flip_fitness((min(fitness)))
        if best_fit > best_fit_iter:
            best_member = pop[fitness.index(max(fitness))]
            best_fit = best_fit_iter
        loss_stats(i, best_fit, best_fit_iter, worst_fit_iter, model_name, texture_type)
        new_pop = [p for _, p in sorted(zip(fitness, pop), reverse=True)][:leave_val]
        fitness_2 = [f**2 for f in fitness]  # we want to make the difference bigger
        total = sum(fitness_2)
        cum_sum = [sum(fitness_2[:i]) / total for i in range(len(fitness))]
        for _ in range(len(pop) - leave_val):
            index1, index2 = find_next(cum_sum), find_next(cum_sum)
            rec = recombinant(pop[index1], pop[index2], fitness_2[index1], fitness_2[index2])
            new_pop.append(mutate(rec, feature_space, threshold_mutation))
        pop = new_pop
    return best_member, best_fit


# def iterate(model_type, feature_space, train_x, test_x, train_y, test_y, test_df, texture_cols,
#             pop_size, permutation_function, num_iter, model_name, new_pop_bol, leave_val, texture_type,
#             threshold_mutation):
#     fitness = []
#     if new_pop_bol:
#         pop = create_pop(feature_space, permutation_function, pop_size)
#     else:
#         pop, fitness = load_pop(model_name, texture_type)
#         if len(pop) < pop_size:
#             pop += create_pop(feature_space, permutation_function, pop_size-len(pop))
#             fitness += (pop_size-len(pop))*[0.0001]
#         elif len(pop) > pop_size:
#             pop = pop[:pop_size]
#             fitness = fitness[:pop_size]
#     best_fit = 100
#     best_member = {}
#     # try:
#     for i in range(num_iter):
#         if new_pop_bol:
#             fitness = [calc_fitness(p, train_x, test_x, train_y, test_y, model_type, test_df, texture_cols)
#                        for p in pop]
#         else:
#             new_pop_bol = True
#         print(fitness)
#         new_pop = [p for _, p in sorted(zip(fitness, pop), reverse=True)][:leave_val]
#         save_top_5(new_pop, fitness[:leave_val], model_name, texture_type)
#         if best_fit > flip_fitness(max(fitness)):
#             best_member = pop[fitness.index(max(fitness))]
#             best_fit = flip_fitness(max(fitness))
#         if i % 1 == 0:
#             print(sorted([flip_fitness(f) for f in fitness]))
#             print_best_loss(i, best_fit, flip_fitness(max(fitness)))
#         total = sum(fitness)
#         cum_sum = [sum(fitness[:i]) / total for i in range(len(fitness))]
#         fitness_3 = [f**1.5 for f in fitness]
#         for _ in range(len(pop) - leave_val):
#             index1, index2 = find_next(cum_sum), find_next(cum_sum)
#             rec = recombinant(pop[index1], pop[index2], fitness_3[index1], fitness_3[index2])
#             new_pop.append(mutate(rec, feature_space, threshold_mutation))
#         pop = new_pop
#     # except Exception as e:
#     #     print(e)
#     #     print(fitness)
#     save_pop(pop, model_name, texture_type, fitness)
#     return best_member, best_fit


def load_pop(model_name, texture_type):
    with open('results_all_models/{0}/pop-{1}'.format(model_name, texture_type), 'rb') as f:
        pop = pickle.load(f)
    with open('results_all_models/{0}/fitness-{1}'.format(model_name, texture_type), 'rb') as f:
        fitness = pickle.load(f)
    return pop, fitness


def save_pop(pop, model_name, texture_type, fitness):
    with open('results_all_models/{0}/pop-{1}'.format(model_name, texture_type), 'wb') as f:
        pickle.dump(pop, f)
    with open('results_all_models/{0}/fitness-{1}'.format(model_name, texture_type), 'wb') as f:
        pickle.dump(fitness, f)

