"""
Test:
Local test on train.py
python train.py --train "../../test_data/train/" --validation "../../test_data/val/" --model-dir "../../test_data/"

vscode launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${fileDirname}",
            "args": [
                "--train",
                "../../test_data/train/",
                "--validation",
                "../../test_data/val/",
                "--model-dir",
                "../../test_data/"
            ]
        }
    ]
}

"""
from sklearn.datasets import make_classification
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import os
from utils import print_files_in_path
import torchaudio


class MyDataset(Dataset):
    def __init__(self, n_samples, n_features, n_classes):
        self.n_samples = n_samples
        self.X, self.Y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            n_classes=n_classes,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, x):
        # Model expect float32
        return torch.tensor(self.X[x, :], dtype=torch.float32), torch.tensor(self.Y[x], dtype=torch.long)


class Net(nn.Module):
    def __init__(self, input_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_features, 3)

    def forward(self, x):
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target.long(), reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


# Sagemaker
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


# Sagemaker
def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)


def main(args):
    """
    SM_CHANNEL does not contain backward slash:
        SM_CHANNEL_TRAIN=/opt/ml/input/data/train
        SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation

    Training job name:
        script-mode-container-xgb-2020-08-10-13-29-15-756

    """
    train_channel, validation_channel, model_dir = args.train, args.validation, args.model_dir

    print("\nList of files in train channel: ")
    print_files_in_path(train_channel)

    print("\nList of files in validation channel: ")
    print_files_in_path(validation_channel)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)
    kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    input_features = 5
    n_samples = 1000
    dataset = MyDataset(n_samples, input_features, 3)
    train_len = int(n_samples * 0.7)
    test_len = n_samples - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Net(input_features).to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # scheduler.step()

    test(model, device, test_loader)

    if args.save_model:
        save_model(model, model_dir)


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--validation", type=str, default=os.getenv("SM_CHANNEL_VALIDATION", None))
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", None))

    args = parser.parse_args()
    print(args)
    main(args)
