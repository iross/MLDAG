import os
import json
import yaml
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'metadata.yaml'), 'r') as file:
    metadata = yaml.safe_load(file)

# wandb login key
os.environ['WANDB_API_KEY'] = metadata['WANDB_API_KEY']

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data splitting
with open(os.path.join(script_dir, metadata['DATA_PATHNAME']), 'r') as f:
    logs = json.load(f)

# create training set
partition_idx = int(len(logs) * 0.80)
seqs = [item[0] for item in logs]
labels = [0 if item[1] == 'transient' else 1 for item in logs] # 0=transient, 1=non-transient
x_train = torch.tensor(seqs[:partition_idx], dtype=torch.float32)
y_train = torch.tensor(labels[:partition_idx], dtype=torch.int)

# create validate and test set via round robin
x_validate, y_validate, x_test, y_test = [], [], [], []
for i, (seq, label) in enumerate(logs[partition_idx + 1:]):
    label = 0 if label == 'transient' else 1
    if i % 2 == 0:
        x_validate.append(seq)
        y_validate.append(label)
    else:
        x_test.append(seq)
        y_test.append(label)
x_validate = torch.as_tensor(x_validate, dtype=torch.float32)
y_validate = torch.as_tensor(y_validate, dtype=torch.int)
x_test = torch.as_tensor(x_test, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.int)


class EarlyStopping:
    """Class for early stopping the process of training the model."""
    def __init__(self, patience_threshold=5, verbose=False, cp_path='checkpoints/default.pt'):
        self.verbose = verbose
        self.patience = 0
        self.patience_threshold = patience_threshold
        self.best_loss = None
        self.early_stop = False
        self.cp_path = cp_path

    def __call__(self, loss, model):
        if self.best_loss is None or loss < self.best_loss:
            self.patience = 0
            self.best_loss = loss
            torch.save(model, os.path.join(script_dir, self.cp_path))
        else:
            self.patience += 1
            if self.verbose:
                print(f'EarlyStopping patience: {self.patience} out of {self.patience_threshold}')
            if self.patience >= self.patience_threshold:
                self.early_stop = True

        return self.early_stop


class LSTMNet(nn.Module):
    """Define the LSTM network"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Get the last time step output
        out = self.sigmoid(out)
        
        return out


def train(config, checkpoint_pathname=None):
    # model hyperparameters
    input_size = 46      # Number of features
    num_classes = 1      # Number of output classes (for binary classification)
    num_layers = config['lstm_layers']      # Number of LSTM layers
    hidden_size = config['hidden_size']     # Number of hidden units

    # training hyperparameters
    num_epochs = config['epochs']           # Number of epochs
    batch_size = config['batch_size']       # Batch size
    learning_rate = config['learning_rate'] # Learning rate

    # create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # initialize the model, loss function, and optimizer
    model = LSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # early stopping for optimizer
    early_stopping = EarlyStopping(patience_threshold=5, verbose=True, cp_path=checkpoint_pathname)

    # train the model
    for epoch in range(num_epochs):
        training_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            
            # forward + optimization
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # post epoch procedures
        training_loss /= len(train_loader.dataset) # normalize loss  
        if early_stopping(training_loss, model): break  
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {training_loss}')
        
        # validation
        validate_dataset = TensorDataset(x_validate, y_validate)
        validate_loader = DataLoader(dataset=validate_dataset, batch_size=256)
        validation_criterion = nn.BCELoss()
        with torch.no_grad():
            validate_loss = 0.0
            for sequences, target in validate_loader:
                sequences = sequences.to(device)
                target = target.to(device).float()
                outputs = model(sequences)
                validate_loss += validation_criterion(outputs.squeeze(), target)
            validate_loss /= len(validate_loader.dataset)
            print(f'Validate Loss: {validate_loss}\n')

        if wandb.run is not None: wandb.log({"validate_loss": validate_loss}) # report to wandb

    print('Training finished.')
    return model


def sweep_model(config=None):
    """Trains the model with wandb integration"""
    with wandb.init(config=config) as run:
        config = run.config
        checkpoint_pathname = f'checkpoints/{run.project}_{run.sweep_id}_{run.id}.pt'
        train(config, checkpoint_pathname)

model_pathname = None
if metadata['WANDB_ENABLE']:   
    wandb.login()

    with open(os.path.join(script_dir, 'sweep_config.yaml'), 'r') as file:
        sweep_config = yaml.safe_load(file)

    # start the sweep
    sweep_id = wandb.sweep(sweep_config, project=metadata['WANDB_PROJECT'])
    wandb.agent(sweep_id, function=sweep_model, count=metadata['WANDB_SWEEP_RUNS'])
    
    # retrieve best run
    sweep = wandb.Api().sweep(f"{metadata['WANDB_ENTITY']}/{metadata['WANDB_PROJECT']}/{sweep_id}")
    best_run = min(sweep.runs, key=lambda run: run.summary.get('validate_loss', float('inf')))
    model_pathname = f"checkpoints/{metadata['WANDB_PROJECT']}_{sweep_id}_{best_run.id}.pt"
    model = torch.load(os.path.join(script_dir, model_pathname))
else:
    model_pathname = 'checkpoints/default.pt'
    with open(os.path.join(script_dir, 'hyperparameters.yaml'), 'r') as file:
        hyperparams = yaml.safe_load(file)
    model = train(hyperparams)


model.eval()  # Set the model to evaluation mode

# Create DataLoader
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=256)

# Testing phase
with torch.no_grad():

    # model testing
    correct = 0
    for data, target in test_loader:
        outputs = model(data)
        predicted = (outputs > 0.5).float().squeeze()
        for i in range(len(predicted)):
            if predicted[i] == target[i]:
                correct += 1
    
    print('Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


print(f'model saved at: {model_pathname}')
