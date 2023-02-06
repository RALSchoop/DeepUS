"""Train a new network."""

import deepus

import torch
import torchvision
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

import coach
import math
import os.path
import time
import datetime

from torchinfo import summary

# Specify the root dataset folder here.
dataset_root = r'/export/scratch2/home/rals/Data/DeepUS/TrainingData/CIRS073_RUMC'

if torch.cuda.is_available():
    c_device = torch.device('cuda')
else:
    c_device = torch.device('cpu')
print(f'Chosen device: {c_device}')

# Writer outputs to ./runs/ by default.
# To view: start tensorboard on command line with: tensorboard --logdir=runs
# Make sure your current directory is on the right drive.
writer = SummaryWriter('runs/deepus_experiment_n')

# On reproducibility: https://pytorch.org/docs/stable/notes/randomness.html.
torch.manual_seed(seed := 1)
torch.backends.cudnn.deterministic = True

# IMPORTANT, to have all tensors of the same dtype.
# More important it seems that cuFFT doesn't work with dtype float64
torch.set_default_dtype(torch.float32)

deepus_dataset = deepus.UltrasoundDataset(
        dataset_root,
        input_transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()]),
        target_transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()]))
h_data = deepus_dataset.load_header()

train_sampler, test_sampler = deepus_dataset.create_samplers(0.2)
# Set the batch size.
train_loader = torch.utils.data.DataLoader(deepus_dataset,
                                           batch_size=(batch_size := 1),
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(deepus_dataset,
                                          batch_size=batch_size,
                                          sampler=test_sampler)
# Examplatory sample.
train_iter = iter(train_loader)
inputs, targets = train_iter.next()

# Declare model
model = deepus.DataFKImageNetwork(h_data, residual=True, num_blocks=3)
model.to(c_device)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), learning_rate := 1e-2,
                             amsgrad=True)

# Log settings about run.
training_settings_path = os.path.join(writer.get_logdir(),
                                      f'training_settings.txt')
with open(training_settings_path, 'w') as logfile:
    logfile.write(str(model))
    logfile.write('\n\n')
    logfile.write(str(summary(model, input_data=inputs, verbose=0, depth=5,
                  row_settings=["depth", "var_names"],
                  col_names=["input_size", "kernel_size", "output_size",
                             "num_params", "mult_adds"])))
    logfile.write('\n\n')
    logfile.write(str(optimizer))
    logfile.write('\n\n')
    logfile.write(f'Train loader size: {len(train_loader)}\n')
    logfile.write(f'Test loader size: {len(test_loader)}\n')
    logfile.write(f'Batch size: {batch_size}\n')

# Single model pass:
# once = model(inputs.to(device=c_device))

best_test_loss = 1_000_000.
training_time_start = time.time()
for epoch in range(epochs := 70):
    epoch_time_start = time.time()
    print(f'\nEpoch: {epoch+1}')

    print() # Just for \n.
    train_loss = coach.train_single_epoch(
        epoch, math.floor(len(train_loader) / 6), train_loader, optimizer,
        model, loss_fn, writer, device=c_device)
    print()
    test_loss = coach.test_single_epoch(
        epoch, math.floor(len(test_loader) / 6), test_loader, model, loss_fn,
        writer, device=c_device)
    epoch_time_end = time.time()
    print(f'\nEpoch Duration: {str(datetime.timedelta(seconds=epoch_time_end - epoch_time_start))}')

    # Log for each epoch the latest per batch loss of training and test.
    writer.add_scalars('Training and Test Loss',
                       {'Training': train_loss, 'Test': test_loss}, epoch + 1)
    writer.flush()

    # Track best performance, and save.
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        model_path = os.path.join(writer.get_logdir(), f'model_epoch{epoch}.msd')
        # Maybe improve this naming, or keep a seperate file of the parameters.
        torch.save(model.state_dict(), model_path)
        print(f'\nImproved test loss, model saved!')

training_time_end = time.time()
print(f'\nTraining time: {str(datetime.timedelta(seconds=training_time_end - training_time_start))}')
