import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Network(torch.nn.Module):
    def __init__(self):
        # initialize the superclass
        super().__init__()

        # hidden layer has 128 nodes and takes in all the inputs

        # the hidden layer will apply a linear transformation W: R^784 -> R^128 and then add B (WX + B)
        # W is a matrix of the weights for each node; It is the stacked transposed weight vector for each node, and B is an R^128 column
        self.hidden_layer = torch.nn.Linear(28 * 28, 128)

        # the output layer takes in all the values from the hidden nodes and outputs the neural network's computed values
        self.output_layer = torch.nn.Linear(128, 10)

        self.activation_fn = torch.nn.ReLU()

    def forward(self, x):
        # each row in x is an image (R^784 vector)
        # whereas calling the hidden layer on an image means to apply it to the inputs of the image, calling it on x means to apply it to each image in the batch
        activation = self.activation_fn(self.hidden_layer(x))
        # output layer does not get passed through an activation function
        out = self.output_layer(activation)
        return out

if __name__ == "__main__":
    device = torch.device("cpu")

    # 10 classes for each digit it could be
    num_classes = 10

    # MNIST uses 28x28 images which means a 28 * 28 vector is the input
    input_size = 28 * 28

    model = Network().to(device)

    # TODO: Study cross entropy
    loss_fn = torch.nn.CrossEntropyLoss()

    # an optimizer contains learning params
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # switch to training mode
    model.train()

    # complete forwards and backwards propogation 10 times to tweak the network
    epochs = 10

    # the transform will convert the images into a tensor, then flatten it to be a 784 node input layer
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    # download MNIST dataset to ./data
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # choose 64 images for the dataset
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("Begun training")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_of_inputs, correct_answers in train_loader:
            # batch of inputs contains a list of images: [0: Image1, 1: Image2, ..., <BATCH_SIZE>: Image<BATCH_SIZE>]
            # correct answers contains the correct classification parallel to the batch of inputs: [0: 3, 1: 9, ..., <BATCH_SIZE>: 0]
            # e.g. from this example, Image1 should be classified as a 3, Image2 should be classified as a 9, and so on

            # move data to CPU
            batch_of_inputs = batch_of_inputs.to(device) 
            correct_answers = correct_answers.to(device)

            # logits are the scores for each class (the digits from 0 - 9) and the one with the highest score is
            # what the model thinks is the correct classification
            logits = model(batch_of_inputs)

            # pass the scores into the loss function along with the correct answer to compute the loss
            loss = loss_fn(logits, correct_answers)

            # perform the backward pass
            # zero the gradients so they dont aggregate
            optimizer.zero_grad()
            # compute gradients for gradient descent
            loss.backward()
            # learn
            optimizer.step()

            # track the total loss by converting loss from a scalar tensor into just a number
            total_loss += loss.item()

            # look through the classes (dim = 0 is batch, dim = 1 is the classes) and get the ones with the highest scores
            preds = torch.argmax(logits, dim=1)

            # add up where both are the same
            correct += (preds == correct_answers).sum().item()

            # add how many batches were just tested
            total += correct_answers.size(0)

        avg_loss = total_loss / len(train_loader)

        # show the steps for accuracy
        accuracy = correct / total
        percent = accuracy * 100

        print(f"Epoch {epoch}/{epochs} | avg loss = {avg_loss:.4f} | accuracy = {percent:.2f}%")

    test_dataset = datasets.MNIST(
        root = "./data",
        train = True,
        transform = transform
    )

    test_loader = DataLoader(test_dataset, 64)


    # Test accuracy again
    model.eval()

    total = 0
    correct = 0

    i = 1
    for batch_of_inputs, correct_answers in test_loader:
        with torch.no_grad():
            logits = model(batch_of_inputs)
            
        preds = torch.argmax(logits, dim = 1)

        correct_iteration = (preds == correct_answers).sum().item()
        total_iteration = correct_answers.size(0)

        correct += correct_iteration
        total += total_iteration

        print(f"{i}) Correct: {correct_iteration}, Incorrect: {total_iteration - correct_iteration}")

        i += 1

    print(f"Accuracy: {correct / total * 100 :.2f}%")

    # Save weights and biases
    torch.save(model.state_dict(), "mnist.pth")