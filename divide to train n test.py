import pickle
import random


def divide(num_of_data_points, num_of_test):
    test_nums = random.sample(range(1, num_of_data_points), num_of_test)
    train_nums = list(set(range(1, num_of_data_points+1)) - set(test_nums))
    with open('test nums', 'wb') as f:
        pickle.dump(test_nums, f)
    with open('train nums', 'wb') as f:
        pickle.dump(train_nums, f)


def main():
    number_of_points = 63
    number_of_test = 9
    divide(number_of_points, number_of_test)


main()








