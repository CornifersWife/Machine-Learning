import csv

from Trainer import Trainer


def read_csv(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        all_data = []
        for row in csv_reader:
            row_data = [float(element) for element in row[:-1]]
            row_data += [row[-1]]
            all_data.append(row_data)
    return all_data


def list_outputs(data):
    unique_values = set()

    for sublist in data:
        if sublist:
            unique_values.add(sublist[-1])

    return list(unique_values)


def signum_fnc(net):
    return 1 if net >= 0 else 0


train_set = read_csv("data/perceptron.data")
test_set = read_csv("data/perceptron.test.data")

learning_rate = 0.01
target_error = 0.01
activation_function = signum_fnc

arr = list_outputs(train_set)

for i in range(len(arr)):
    print(str(i + 1) + '. ' + arr[i])

desired_i = int(input(f'Choose desired outputs:  ')) - 1

trainer = Trainer(learning_rate, target_error, train_set, test_set)
trainer.initialize_perceptron(arr[desired_i], activation_function)

trainer.teach()

example_vector = train_set[0]
formatted_vector = ", ".join([f"{elem}" for elem in example_vector])
print(f"Input singular vector written in like this:   (write  stop  to terminate this program)\n{formatted_vector}")

while True:
    input_vector = input()
    if input_vector == "stop":
        break
    input_vector = input_vector.split(',')
    try:

        input_vector[:-1] = [float(x.strip()) for x in input_vector[:-1]]
        output = "not " if trainer.perceptron.compute_out(input_vector[:-1]) == 0 else " "
        output += trainer.perceptron.desired
        print(f'For this vector the program has guessed it is {output}')
    except Exception:
        print('Incorrect data input, try again')
