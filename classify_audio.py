import torch
import torchaudio
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
from torchviz import make_dot

import matplotlib.pyplot as plt
import time


class CNN_ReLU(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(7680, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        # predictions = self.softmax(logits)
        return logits

    def get_loss(self, learning_rate):
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return loss, optimizer


class CNN_LeakyReLU(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(7680, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        # predictions = self.softmax(logits)
        return logits

    def get_loss(self, learning_rate):
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return loss, optimizer


class CNN_Tanh(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(7680, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        # predictions = self.softmax(logits)
        return logits

    def get_loss(self, learning_rate):
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return loss, optimizer


class CNN_Sigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(7680, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        # predictions = self.softmax(logits)
        return logits

    def get_loss(self, learning_rate):
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return loss, optimizer


def visualize_cnn(cnn, device, input_size):
    fake_input = Variable(
        torch.zeros((1, input_size[0], input_size[1], input_size[2]))
    ).to(device)
    outputs = cnn(fake_input)
    return make_dot(outputs, dict(cnn.named_parameters()))


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate, n_fft=1024, hop_length=512, n_mels=64):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    def forward(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-10)

        return log_mel_spec


def split_data(dataset, train_split=0.7, val_split=0.15, test_split=0.15):
    assert (
        abs(train_split + val_split + test_split - 1.0) < 1e-6
    ), "Splits must sum to 1"
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset


def create_data_loader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size)
    return dataloader


def train_cnn(
    cnn, train_loader, val_loader, test_loader, n_epochs, learning_rate, device
):
    """
    Train a the specified network.

        Outputs a tuple with the following five elements
        train_index
        train_losses
        val_index
        val_losses
        accuracy
    """
    loss, optimizer = cnn.get_loss(learning_rate)
    print_every = 32
    idx = 0

    train_index = []
    train_losses = []
    val_index = []
    val_losses = []
    accuracies = []

    training_start_time = time.time()

    for epoch in range(n_epochs):
        running_loss = 0.0
        start_time = time.time()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            # reset optimizer gradient
            optimizer.zero_grad()

            # forward pass
            outputs = cnn(inputs)

            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # update stats
            running_loss += loss_size.data.item()

            # print every nth batch of an epoch
            if (i % print_every) == print_every - 1:
                print(
                    f"Epoch {epoch + 1}, Iteration {i + 1}\t train_loss: {running_loss / print_every:.2f} took: {time.time() - start_time:.2f}s"
                )
                # Reset running loss and time
                train_losses.append(running_loss / print_every)
                train_index.append(idx)
                running_loss = 0.0
                start_time = time.time()
            idx += 1

        # validation pass at the end of each epoch
        total_val_loss = 0
        for inputs, labels in val_loader:
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            # forward pass
            val_outputs = cnn(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()
        val_losses.append(total_val_loss / len(test_loader))
        val_index.append(idx)
        print(f"Validation loss = {total_val_loss / len(test_loader):.2f}")

        # test pass at the end of each epoch
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient calculation for validation/testing
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # forward pass
                test_outputs = cnn(inputs)

                # get predictions (index of the max logit)
                _, predicted = torch.max(test_outputs, 1)

                # update total and correct predictions count
                total += labels.size(0)  # total number of samples
                correct += (
                    (predicted == labels).sum().item()
                )  # count of correct predictions

        # calculate and store accuracy for this epoch
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Accuracy = {accuracy:.2f}")

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    return {
        "train_index": train_index,
        "train_losses": train_losses,
        "val_index": val_index,
        "val_losses": val_losses,
        "accuracies": accuracies,
    }


LABEL_ID = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}


def plot_mel_spectrograms_by_label(mel_specs_by_label):
    fig, axs = plt.subplots(
        10, 4, figsize=(30, 25)
    )  # 10 rows for 10 labels, 4 columns for each occurrence
    fig.suptitle(
        "Mel Spectrograms by Truth Label", fontsize=16
    )  # Main title for the entire figure

    for label, mel_specs in mel_specs_by_label.items():
        for i, (mel_spec, ax) in enumerate(zip(mel_specs, axs[label])):
            mel_spec = (
                mel_spec.squeeze().cpu().numpy()
            )  # Remove batch and channel dimensions
            img = ax.imshow(
                mel_spec, interpolation="nearest", aspect="auto", origin="lower"
            )
            ax.set_title(f"{LABEL_ID[label]}, Occurrence {i+1}")
            ax.set_xlabel("Time (frames)")
            ax.set_ylabel("Mel Filter Bins (Logscale)")
            fig.colorbar(img, ax=ax)  # Add colorbar to each subplot

    plt.tight_layout()
    plt.show()
