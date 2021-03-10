import random


def create_pop(feature_len, pop_size=30):
    return [[0 if random.random() > 0.5 else 1 for _ in range(feature_len)] for _ in range(pop_size)]


def recombinant(p1, p2, fit1, fit2):
    threshold = fit1 / (fit1 + fit2)
    return [p1[i] if random.random() > threshold else p2[i] for i in range(len(p1))]


def mutate(p1):
    threshold = 0.1
    return [item if random.random() > threshold else (1 - item) for item in p1]


def calc_fitness(p, model_type, train_df, test_df, train_y, test_y, all_cols_for_model, length_penalty):
    cols_for_model = [all_cols_for_model[i] for i in range(len(all_cols_for_model)) if p[i] == 1]
    train_x = train_df[cols_for_model]
    test_x = test_df[cols_for_model]
    model = model_type(train_x, test_x, train_y, test_y)
    model.create_model()
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


def print_best_loss(i, best_loss, best_member):
    print("i = {0}, best loss = {1}".format(i, best_loss))  # because we wnt to see the actual loss
    print("num of features = {0}".format(sum(best_member)))
    print()


def iterate(model, num_iter, feature_len, pop_size, train_df, test_df, texture_cols, cols_for_model, length_penalty):
    pop = create_pop(feature_len, pop_size)
    leave_val = int(pop_size * 0.2)
    train_y, test_y = train_df[texture_cols], test_df[texture_cols]
    best_fit = 0
    best_member = [1]*feature_len
    for i in range(num_iter):
        fitness = [calc_fitness(p, model, train_df, test_df, train_y, test_y, cols_for_model, length_penalty)
                   for p in pop]
        if to_loss(best_fit, best_member, length_penalty) > to_loss(max(fitness), pop[fitness.index(max(fitness))],
                                                                    length_penalty):
            best_fit = max(fitness)
            best_member = pop[fitness.index(best_fit)]
        if i % 5 == 0:
            print_best_loss(i, to_loss(best_fit, best_member, length_penalty), best_member)
        total = sum(fitness)
        cum_sum = [sum(fitness[:i]) / total for i in range(len(fitness))]
        new_pop = [p for _, p in sorted(zip(fitness, pop), reverse=True)][:leave_val]
        for _ in range(len(pop) - leave_val):
            index1, index2 = find_next(cum_sum), find_next(cum_sum)
            rec = recombinant(pop[index1], pop[index2], fitness[index1], fitness[index2])
            new_pop.append(mutate(rec))
        pop = new_pop
    return best_member, to_loss(best_fit, best_member, length_penalty)  # because we wnt to see the actual loss

