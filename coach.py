"""Helpful functionality for training."""

import deepus
import torch
import matplotlib.pyplot as plt
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

def test_single_epoch(epoch_idx: int, 
                      avg_batch_amount: int,
                      test_loader: torch.utils.data.DataLoader,
                      model: torch.nn.Module,
                      loss_fn: torch.nn.modules.loss._Loss,
                      writer: SummaryWriter,
                      device=None
                      ) -> float:
    """Single pass over the test set.

    Arguments:
     - epoch_idx: Index of the current epoch (0-based).
     - avg_batch_amount: The amount of batches over which to calculate
        the average loss.
     - test_loader: Testing dataloader.
     - model: Model to infer.
     - loss_fn: Loss function.
     - writer: Tensorboard SummaryWriter instance for logging.

    Returns:
     - The total average loss of the epoch.
    """

    inter_loss_accumulator = 0.
    total_loss_accumulator = 0.
    last_avg_loss = 0.

    # No need of training mode and gradients.
    model.train(False)
    with torch.no_grad():
        for batch_idx, samples in enumerate(test_loader):
            inputs, targets = samples
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets.to(torch.get_default_dtype()))

            inter_loss_accumulator += loss.item()
            total_loss_accumulator += loss.item()
            if batch_idx % avg_batch_amount == avg_batch_amount - 1:
                last_avg_loss = inter_loss_accumulator / avg_batch_amount
                print(f'Test Batch {batch_idx + 1}, Latest Average Test Loss:'
                      f' {last_avg_loss}')
                axis_value = epoch_idx * len(test_loader) + batch_idx + 1
                writer.add_scalar('Test loss', last_avg_loss, axis_value)
                writer.flush()
                # Reset accumulator for next chunk.
                inter_loss_accumulator = 0.
        total_avg_loss = total_loss_accumulator / len(test_loader)
        print(f'Total Average Test Loss: {total_avg_loss}')
    return total_avg_loss

def train_single_epoch(epoch_idx: int,
                       avg_batch_amount: int,
                       train_loader: torch.utils.data.DataLoader,
                       optimizer: torch.optim.Optimizer,
                       model: torch.nn.Module,
                       loss_fn: torch.nn.modules.loss._Loss,
                       writer: SummaryWriter,
                       device = None
                       ) -> float:
    """Trains the model for a single pass over the whole train loader.

    Arguments:
     - epoch_idx: Index of the current epoch (0-based).
     - avg_batch_amount: The amount of batches over which to calculate
        the average loss.
     - train_loader: Training dataloader.
     - optimizer: Optimizer for training.
     - model: Model to train.
     - loss_fn: Loss function.
     - writer: Tensorboard SummaryWriter instance for logging.

    Returns:
     - The last average loss of the epoch.
    """
    model.train(True)

    loss_accumulator = 0.
    last_avg_loss = 0.

    for batch_idx, sample in enumerate(train_loader):
        inputs, targets = sample
        inputs = inputs.to(device=device)
        targets = targets.to(device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets.to(torch.get_default_dtype()))
        loss.backward()
        optimizer.step()
        
        loss_accumulator += loss.item()
        if batch_idx % avg_batch_amount == avg_batch_amount - 1:
            # The average per batch.
            last_avg_loss = loss_accumulator / avg_batch_amount
            print(f'Training Batch {batch_idx + 1}, Latest Average Train Loss:'
                  f' {last_avg_loss}')
            axis_value = epoch_idx * len(train_loader) + batch_idx + 1
            writer.add_scalar('Training loss', last_avg_loss, axis_value)
            writer.flush()
            # Reset accumulator for next chunk.
            loss_accumulator = 0.
    
    # Return loss of latest average amount of the epoch.
    return last_avg_loss

def generate_batch_display(data, targets, h_data):
    """Create matplotlib.pyplot.figure instances for displaying the batch.

    Three image sets are generated:
     - Data is plotted as a graph.
     - Data is migrated and post-processed to create a manual reconstructed
        image set.
     - Target images.
    
    Returns tuple of size 3 with each entry being a list with either
    plotted data, migrated data image or target image.
    """
    input_imgs = deepus.torch_image_formation_batch(torch.real(
        deepus.torch_fkmig_batch(data, h_data)))

    datafigs, input_img_figs, targetfigs = [], [], []
    for batch in range(data.size(0)):
        datafig = plt.figure(batch)
        plt.plot(range(torch.squeeze(data, 1)[batch, :, :].numpy().shape[0]),
                torch.squeeze(data, 1)[batch, :, :].numpy()[:, 64])
        datafigs.append(datafig)
        plt.close(datafig)

        input_img_fig = plt.figure(batch + data.size(0))
        plt.imshow(torch.squeeze(
            input_imgs, 1)[batch, :h_data['nz_cutoff'], :].numpy(),
            aspect='auto', cmap='gray')
        input_img_figs.append(input_img_fig)
        plt.close(input_img_fig)

        targetfig = plt.figure(batch + 2 * data.size(0))
        plt.imshow(torch.squeeze(
            targets, 1)[batch, :h_data['nz_cutoff'], :].numpy(),
            aspect='auto', cmap='gray')
        targetfigs.append(targetfig)
        plt.close(targetfig)
    
    return datafigs, input_img_figs, targetfigs