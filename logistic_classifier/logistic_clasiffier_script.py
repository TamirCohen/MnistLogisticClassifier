# %%
from tokenize import String
import numpy as np
import logging
import gzip
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy.typing as npt


PWD="/home/duzicman/projects/tau/statistical_machine_learning/logistic_classifier/datasets"
DATASET = {"t_labels": {"path" :PWD + "/train-labels-idx1-ubyte.gz", "offset":8}, "x_samples": {"path": PWD + "/train-images-idx3-ubyte.gz", "offset":16}}
def show_image(image):
    pixels = np.reshape(image, (IMAGE_LENGTH, IMAGE_LENGTH))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def one_hot_encoding(labels):
    label_ammount = np.max(labels) + 1
    return np.eye(label_ammount)[labels]

@dataclass
class LabeledDataSet:
    t_labels: npt.ArrayLike
    x_samples: npt.ArrayLike

    def split(self, sizes):
        indices = np.random.permutation(self.x_samples.shape[1])
        set_indices = {}
        index_offset = 0
        data_sets = {}
        for set_name, set_size in sizes.items():
            set_indices[set_name] = indices[index_offset:index_offset +int(len(indices) * set_size)]
            index_offset += int(len(indices) * set_size)
            data_sets[set_name] = LabeledDataSet(x_samples=self.x_samples[:, set_indices[set_name]], t_labels=self.t_labels[set_indices[set_name], :])
        return data_sets

class MnistLogisticData():
    IMAGE_LENGTH = 28
    IMAGE_SIZE = IMAGE_LENGTH ** 2
    TRAINING_SET_SIZE = 0.6
    VALIDATION_SET_SIZE = 0.2
    TEST_SET_SIZE = 0.2

    def __init__(self, data_sets_paths, set_sizes=None):
        self.data_sets_paths = data_sets_paths
        self.data_sets = {}
        if set_sizes is None:
            self.set_sizes = {"training":self.TRAINING_SET_SIZE, "validation": self.VALIDATION_SET_SIZE, "test": self.TEST_SET_SIZE}
        else:
            self.set_sizes = set_sizes

    def load_dataset(self, file_path: str, offset: int):
        with gzip.open(file_path, 'rb') as data_file:
            data = np.frombuffer(data_file.read(), np.uint8, offset=offset)
        return data
    
    def create_data_sets(self):
        x_samples = self.load_dataset(self.data_sets_paths["x_samples"]["path"], self.data_sets_paths["x_samples"]["offset"])
        t_labels = self.load_dataset(self.data_sets_paths["t_labels"]["path"], self.data_sets_paths["t_labels"]["offset"])
        data_set = LabeledDataSet(t_labels=t_labels, x_samples=x_samples)


        data_set.t_labels = one_hot_encoding(data_set.t_labels)
        data_set.x_samples = np.reshape(data_set.x_samples, (-1, self.IMAGE_SIZE))
        data_set.x_samples = np.c_[data_set.x_samples, np.ones(data_set.x_samples.shape[0])]
        data_set.x_samples = np.transpose(data_set.x_samples)
        self.class_number = data_set.t_labels.shape[1]
        self.data_sets = data_set.split(self.set_sizes)

class MnistLogisticClassifier():
    def __init__(self, data: MnistLogisticData, initial_weights, learning_rate: float, accuracy_epsilon:float) -> None:
        self.data = data
        self.initial_weights = initial_weights
        self.learning_rate = learning_rate
        self.accuracy_epsilon = accuracy_epsilon
        
    @classmethod
    def cross_entropy_loss(cls, y_labels_probabilites, t_labels) -> float:
        natural_log = lambda x: np.log(x)
        y_logged = np.vectorize(natural_log)(y_labels_probabilites)
        
        # Calculating np.trace(np.matmul)) but just calculating the diagonal instead of all the matrix 
        loss = -np.einsum('nk,nk->', t_labels, y_logged)
        return loss

    @classmethod
    def caculate_lables_probabilities(cls, w_weights, x_samples):
        a_logits = np.matmul(np.transpose(x_samples), w_weights)
        exponent = lambda x: np.exp(x)
        exponent_logits = np.vectorize(exponent)(a_logits)
        row_summation = np.sum(exponent_logits, axis=1)[:, np.newaxis]
        y_labels_probabilities = exponent_logits /  row_summation
        return y_labels_probabilities

    @classmethod
    def cross_entropy_loss_gradient(cls, y_label_probabilites, data_set: LabeledDataSet):
        return np.matmul(data_set.x_samples, (y_label_probabilites - data_set.t_labels))

    @classmethod
    def classification_accuracy(cls, y_label_probabilites, t_labels):
        classification = np.eye(y_label_probabilites.shape[0], y_label_probabilites.shape[1])[np.argmax(y_label_probabilites, axis=1)]
        return np.sum(np.all(t_labels == classification, axis=1)) / classification.shape[0]

    @classmethod
    def callisification_matrics(cls, w_weights, data_set: LabeledDataSet):
        y_labels_probabilities = cls.caculate_lables_probabilities(w_weights, data_set.x_samples)
        accuracy = cls.classification_accuracy(y_labels_probabilities, data_set.t_labels)
        loss = cls.cross_entropy_loss(y_labels_probabilities, data_set.t_labels)
        return y_labels_probabilities, accuracy, loss

    def gradient_descent(self):
        w_weights = self.initial_weights
        training_accuracies = []
        training_losses = []
        validation_losses = []
        validation_accuracies = []
        training = self.data.data_sets["training"]
        validation = self.data.data_sets["validation"]
        
        while len(validation_accuracies) < 2 or abs(validation_accuracies[-1] - validation_accuracies[-2]) > self.accuracy_epsilon:
            y_training_labels_probabilities, training_accuracy, training_loss = self.callisification_matrics(w_weights, training)
            y_validation_labels_probabilities, validation_accuracy, validation_loss = self.callisification_matrics(w_weights, validation)
            loss_gradient = self.cross_entropy_loss_gradient(y_training_labels_probabilities, training)
            print(training_loss)
            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
            w_weights -= loss_gradient * self.learning_rate
        
        plt.title("Accuracy")
        plt.plot(range(len(training_accuracies)),training_accuracies, color="red")
        plt.plot(training_accuracies, color="red")
        plt.plot(validation_accuracies, color="blue")
        plt.show()
        plt.plot(training_losses)
        plt.show()
        print("Training Loss {}".format(training_loss))
        print("Training Accuraccy {}".format(training_accuracy))
        print("Validation Loss {}".format(validation_loss))
        print("Validation Accuraccy {}".format(validation_accuracy))
        print("Steps {}".format(len(training_losses)))
# %%


#TODO add typing in each function
def main():
    # %%
    data = MnistLogisticData(DATASET)
    data.create_data_sets()
    # %%
    classifier = MnistLogisticClassifier(data, initial_weights=np.zeros((data.IMAGE_SIZE + 1, data.class_number)), learning_rate=0.000000002, accuracy_epsilon=0.001)
    classifier.gradient_descent()
    # %%
    #TODO show test Accuracy
    
if __name__ == "__main__":
    main()
