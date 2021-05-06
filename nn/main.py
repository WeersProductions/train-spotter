import torch
import torchvision
import torchvision.transforms as transforms
from image_dataset import TrainImageDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os
from model import Net
import torch.optim as optim
import torch.nn as nn


def load_dataset(img_dir, labels_file, test_split=.2, shuffle_dataset=True, random_seed=42, batch_size=16, image_size=128):
    """
    Load the dataset and split in train and test set.
    """
    transform = transforms.Compose(
        [transforms.ConvertImageDtype(torch.float), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((image_size, image_size))
        ])
    dataset = TrainImageDataset(labels_file, img_dir, transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, test_loader, dataset


def to_rgb_image(tensor):
    np_img = tensor.cpu().numpy().transpose((1, 2, 0))
    m1, m2 = np_img.min(axis=(0, 1)), np_img.max(axis=(0, 1))
    return (255.0 * (np_img - m1) / (m2 - m1)).astype("uint8")


def imshow(img):
    # img = img / 2 + 0.5
    # npimg = img
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imshow(img)
    plt.show()


def main(img_dir, labels_file, model_save_file, train=True, load=True, epochs=2, show_images=False):
    print("Starting")
    batch_size = 16
    image_size = 128
    train_loader, test_loader, dataset = load_dataset(img_dir, labels_file, batch_size=batch_size, image_size=image_size)
    classes = dataset.get_classes()

    if show_images:
        dataiter = iter(train_loader)
        images, labels = dataiter.next()

        print(' '.join('%5s' % label for label in labels))
        imshow(to_rgb_image(torchvision.utils.make_grid(images)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = Net(len(classes), image_size)
    if load and os.path.exists(model_save_file):
        print("Loaded existing model!")
        net.load_state_dict(torch.load(model_save_file))
    net.to(device)
    print(net)
    weights = torch.FloatTensor(dataset.get_class_weights()).to(device)
    print(weights)
    criterion = nn.CrossEntropyLoss(weights)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if train:
        # Train
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print stats
                running_loss += loss.item()
                if i % 10 == 9:
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            
            torch.save(net.state_dict(), model_save_file)
    
    # Test
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))

    print('Finished training')


if __name__ == '__main__':
    model_save_file = "./model_net.pth"
    img_dir = "./data/output/"
    labels_file = os.path.join(img_dir, "label_index.parquet")
    main(img_dir, labels_file, model_save_file, train=True, load=True, show_images=True)
