import data
import sys

training_data = data.labelled_data("labeled_data", "training", data.get_transform(train=True))
print(training_data[int(sys.argv[1])])