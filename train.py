"""Train a new network."""

import deepus

import torch
import torchvision
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

import coach
import math
import os.path
from os import makedirs
import time
import datetime

from torchinfo import summary

#### user defined settings #####################################################

# Specify the root dataset folder here.
data_root      = r'/export/scratch2/felix/Dropbox/Data/US/DeepUSData/'
data_set       = 'CIRS073_RUMC'

# For each random seed specified below, the script will train the same network with different 
# randomly chosen data samples and differently randomly initialized weights. This can be used to 
# assess average performance of a particular network model and training data size
random_seeds   = range(1,17)

# choosing a value between 0 and 1 will determine the number of training samples used 
# (1.0 = 176 training samples)
train_fraction = 0.1

# choose 'full' , 'post' or 'pre' (termed "complete", "post-processing", "pre-processing" in the paper)
model_type     = 'full' 

################################################################################

if torch.cuda.is_available():
    gpu_index = int(input("choose the gpu index (default: 0) : ") or "0")
    c_device = torch.device('cuda:{}'.format(gpu_index))
else:
    c_device = torch.device('cpu')
print(f'Chosen device: {c_device}')

for random_seed in random_seeds:

    # Writer outputs to ./runs/ by default.
    # To view: start tensorboard on command line with: tensorboard --logdir=runs
    # Make sure your current directory is on the right drive.
    trained_nn_path = os.path.join(data_root, 'TrainedNetworks', data_set, model_type,
                                   'trainfrac{}'.format(train_fraction), 'rnd{}'.format(random_seed))
    makedirs(trained_nn_path, exist_ok=True)
    writer = SummaryWriter(trained_nn_path)

    # On reproducibility: https://pytorch.org/docs/stable/notes/randomness.html.
    torch.manual_seed(seed := random_seed)
    torch.backends.cudnn.deterministic = True

    # IMPORTANT, to have all tensors of the same dtype.
    # More important it seems that cuFFT doesn't work with dtype float64
    torch.set_default_dtype(torch.float32)

    deepus_dataset = deepus.UltrasoundDataset(
            os.path.join(data_root, 'TrainingData', data_set),
            input_transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()]),
            target_transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()]))
    h_data = deepus_dataset.load_header()

    train_sampler, test_sampler = deepus_dataset.create_samplers(test_fraction  = 0.2,
                                                                 train_fraction = train_fraction)
    # Set the batch size.
    train_loader = torch.utils.data.DataLoader(deepus_dataset,
                                               batch_size=(batch_size := 1),
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(deepus_dataset,
                                              batch_size=batch_size,
                                              sampler=test_sampler)
    # Examplatory sample.
    train_iter = iter(train_loader)
    inputs, targets = next(train_iter)

    # Declare model
    if model_type == 'pre': # pre processing only
        model = deepus.DataFKImageNetwork(h_data, residual=True, cnn_pre=True, cnn_post=False, num_blocks=6)
    elif model_type == 'post': # post processing only
        model = deepus.DataFKImageNetwork(h_data, residual=True, cnn_pre=False, cnn_post=True, num_blocks=6)
    elif model_type == 'full': # full model
        model = deepus.DataFKImageNetwork(h_data, residual=True, cnn_pre=True, cnn_post=True, num_blocks=3)
    else:
        raise ValueError("model_type needs to be 'pre', 'post' or 'full'")

    model.to(c_device)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), learning_rate := 1e-2,
                                 amsgrad=True)


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
            #model_path = os.path.join(writer.get_logdir(), f'model_epoch{epoch}.msd')
            model_path = os.path.join(writer.get_logdir(), 'model_best.msd')
            # Maybe improve this naming, or keep a seperate file of the parameters.
            torch.save(model.state_dict(), model_path)
            print('\nImproved test loss, model saved!')

    training_time_end = time.time()
    print(f'\nTraining time: {str(datetime.timedelta(seconds=training_time_end - training_time_start))}')

    # Log settings about run.
    training_settings_path = os.path.join(writer.get_logdir(), 'training_settings.txt')
    with open(training_settings_path, 'w') as logfile:
        logfile.write(str(model))
        logfile.write('\n\n')
        logfile.write(str(summary(model, input_data=inputs.to(device=c_device), verbose=0, depth=5,
                      row_settings=["depth", "var_names"],
                      col_names=["input_size", "kernel_size", "output_size",
                                 "num_params", "mult_adds"])))
        logfile.write('\n\n')
        logfile.write(str(optimizer))
        logfile.write('\n\n')
        logfile.write(f'Random Seed: {random_seed}\n')
        logfile.write(f'Train loader size: {len(train_loader)}\n')
        logfile.write(f'Test loader size: {len(test_loader)}\n')
        logfile.write(f'Batch size: {batch_size}\n')
        logfile.write(f'\nTraining time: {str(datetime.timedelta(seconds=training_time_end - training_time_start))}')
