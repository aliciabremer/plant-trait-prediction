import numpy as np
import torch
import torchvision
import torch.nn as nn
import torcheval
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataset import PlantData, PlantDataTest
from newmodels import TransferDinoBig, TransferSWINBatch
from scipy.stats import zscore
import torcheval.metrics
import argparse
import timm

'''
Function for training the model
'''
def train(model, e, optimizer, metric, mse_loss, train_dataloader, lr_scheduler):
    # train
    model.train()
    metric.reset()

    # track stats (current_train_r2 not needed)
    current_train_loss = []

    # loop through data
    for batch_idx, (images, extra, targets) in enumerate(train_dataloader):
        images = images.to(device)
        extra = extra.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model.forward(images,extra)
        train_loss = mse_loss(output, targets)
        train_loss.backward()
        optimizer.step()

        current_train_loss.append(train_loss.item()) # save loss
        
        # r2 score
        metric.update(output, targets)

        if batch_idx%100 == 0:
            print(f'Epoch {e+1}: Batch {batch_idx}: Loss {train_loss.item()}, R2 {metric.compute().item()}')
    
    
    lr_scheduler.step()
    print(f'Training Loss {np.array(current_train_loss).mean()} R2 {metric.compute().item()}')
    return np.array(current_train_loss).mean(), metric.compute().item()

'''
Function for testing the model
'''
def test(model, e, metric, mse_loss, val_dataloader):
    # Test
    model.eval()
    metric.reset()
        
    current_val_loss = []
    for _, (images, extra, targets) in enumerate(val_dataloader):
        images = images.to(device)
        extra = extra.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model.forward(images,extra)
        
        test_loss = mse_loss(output, targets)

        current_val_loss.append(test_loss.item()) # save loss

        # r2 score
        metric.update(output, targets)

    print(f'Val Loss: {np.array(current_val_loss).mean()}. R2: {metric.compute().item()}')
    return np.array(current_val_loss).mean(), metric.compute().item()

'''
Function for saving the model and output csv
'''
def save(model, e, output_csv, means, stds, test_dataloader, path):
    for i, (images, extra) in enumerate(test_dataloader):
        images = images.to(device)
        extra = extra.to(device)

        with torch.no_grad():
            output = model.forward(images,extra).cpu().detach()

        output_csv.iloc[i*NUM_TEST:min((i+1)*NUM_TEST, len(test_dataset)), 0:1000] = (output * (stds[-6:]) + means[-6:]).numpy()

    output_csv.to_csv(f'results_during_training/{path}_predictions_epoch{e+1}.csv', index_label='id')
    torch.save(model, f'results_during_training/{path}_model_epoch{e+1}.pth')


if __name__ == '__main__':
    # a variety of possible arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', type=str, default='vit_v3')
    parser.add_argument('--batch', dest='batch_size', type=int, default=64)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001)
    parser.add_argument('--lr-all', dest='learning_rate_all', type=float, default=0.00001)
    parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=10)
    parser.add_argument('--frozen-epochs', dest='frozen_epochs', type=int, default=4)
    parser.add_argument('--decay', dest='decay', type=float, default= 0.00001)
    parser.add_argument('--decay-all', dest='decay', type=float, default= 0.00001)
    parser.add_argument('--dropout-fc', dest='dropout_fc', type=float, default=0.2)
    parser.add_argument('--dropout-t', dest='dropout_t', type=float, default=0.1)
    parser.add_argument('--stochastic-depth', dest='stochastic_depth', type=float, default=0.1)
    parser.add_argument('--early-stop', dest='early_stop', type=int, default=3)
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.05)
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--lr-sched', dest='lr_sched', type=int, default=6)
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    ALL_LR = args.learning_rate_all
    LRDECAY = args.decay
    ALL_LRDECAY = args.decay
    MAX_EPOCHS = args.max_epochs
    FROZEN_EPOCHS = args.frozen_epochs
    NUM_TEST=128
    
    # assume data in data folder
    dir_path = os.path.join('data')
    
    # seed everything
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    # get device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if device == torch.device("cuda:0"):
        print('GPU')
    else:
        print('CPU')
  
    # load train and test data
    data = pd.read_csv(os.path.join(dir_path, 'train.csv'))
    test_data = pd.read_csv(os.path.join(dir_path, 'test.csv'))
    
    # split data into training and validation data
    train_data, val_data = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=False, random_state=7)
        
    means = torch.tensor(train_data.mean(axis=0), dtype=torch.float32)
    stds = torch.tensor(train_data.std(axis=0), dtype=torch.float32)
    
    # data transforms

    data_transform = torchvision.transforms.Lambda(
        lambda x : (x - means[1:-6])/(stds[1:-6])
    )
    
    # target transforms

    target_transforms = torchvision.transforms.Lambda(
        lambda x : (x - means[-6:])/(stds[-6:])
    )

    # image transforms
    image_train_data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandAugment(),
        torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
        torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(1, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(20),
        torchvision.transforms.ColorJitter(0.2, 0.2, 0.2),
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
            lambda x: torch.clamp(x,min=0, max=1)
        ),
        # use imagenet normalization
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        torchvision.transforms.RandomErasing(p=0.2, scale=(0.001, 0.1), ratio=(0.1, 4), value=0)
    ])

    image_test_data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.ToTensor(),
        # use resnet normalization
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # datasets

    train_dataset = PlantData(train_data,
                            os.path.join(dir_path, 'train_images'),
                            image_train_data_transforms,
                            data_transform,
                            target_transforms)
    val_dataset = PlantData(val_data,
                            os.path.join(dir_path,  'train_images'),
                            image_test_data_transforms,
                            data_transform,
                            target_transforms)
    test_dataset = PlantDataTest(test_data,
                                os.path.join(dir_path,  'test_images'),
                                image_test_data_transforms,
                                data_transform)
        
    # dataloaders

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=1,
                                                pin_memory=False)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=1,
                                                pin_memory=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=NUM_TEST,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=1,
                                                pin_memory=False)
    
    if args.type == 'dino_big':
        base_model = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
        model = TransferDinoBig(base_model, dropout_features=args.dropout_fc, dropout_after=args.dropout_fc).to(device)
    elif args.type == 'swin_v2_b_batch':
        base_model  = torchvision.models.swin_v2_b(
            weights=torchvision.models.Swin_V2_B_Weights.DEFAULT,
            **{
                'dropout':args.dropout_t,
                'attention_dropout': args.dropout_t
            }
        ).to(device)
        model = TransferSWINBatch(base_model, dropout_features=args.dropout_fc, dropout_after=args.dropout_fc).to(device)
            
    # loss, optimizer and R2 score
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=LRDECAY)
    metric = torcheval.metrics.R2Score()
    
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sched, gamma=0.1)
    
    # sample submission file
    sample = pd.read_csv(os.path.join(dir_path, 'sample_submission.csv'))
    
    arr = np.zeros((len(test_dataset), 6))
    output_csv = pd.DataFrame(arr, columns=sample.columns[1:])
    output_csv = output_csv.set_index(test_data['id'])
    
    # save metrics
    # val_r2_saved = []
    train_r2_saved = []
    # test_loss_saved = []
    train_loss_saved = []
    val_loss = []
    val_r2 = []
    
    # Track min loss and corresponding epoch for early stopping
    min_val_loss = 1
    min_val_loss_epoch = 0

    epochs_not_improving = 0
    
    # Training with Base Model Weights Frozen
    for e in range(FROZEN_EPOCHS):

        l, r = train(model, e, optimizer, metric, mse_loss, train_dataloader, lr_scheduler)
        train_loss_saved.append(l)
        train_r2_saved.append(r)
        
        vl, vr = test(model, e, metric, mse_loss, val_dataloader)
        val_loss.append(vl)
        val_r2.append(vr)
        
        if vl < min_val_loss:
            min_val_loss = vl
            min_val_loss_epoch = e
            epochs_not_improving = 0
        else:
            epochs_not_improving += 1
        
        save(model, e, output_csv, means, stds, test_dataloader, args.type + '_seed' + str(args.seed))
        
        if epochs_not_improving > args.early_stop:
            break
        
    print(f'Min Val Loss Epoch {min_val_loss_epoch+1}')
        
    # Reload best model (lowest validation loss) and create new learning
    # rate and learning rate scheduler based on learning rate after unfreezing
    # the base model weights
    del model,base_model
    model = torch.load(f'results_during_training/{args.type}_seed{args.seed}_model_epoch{min_val_loss_epoch+1}.pth').to(device)
    model.base_model = model.base_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ALL_LR, weight_decay=ALL_LRDECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sched, gamma=0.1)
        
    # Training with all weights being updated
    for e in range(MAX_EPOCHS):

        l, r = train(model, FROZEN_EPOCHS+e, optimizer, metric, mse_loss, train_dataloader, lr_scheduler)
        train_loss_saved.append(l)
        train_r2_saved.append(r)
        
        vl, vr = test(model, FROZEN_EPOCHS+e, metric, mse_loss, val_dataloader)
        val_loss.append(vl)
        val_r2.append(vr)
        
        if vl < min_val_loss:
            min_val_loss = vl
            min_val_loss_epoch = e
            epochs_not_improving = 0
        else:
            epochs_not_improving += 1
        
        save(model, FROZEN_EPOCHS+e, output_csv, means, stds, test_dataloader, args.type + '_seed' + str(args.seed))

        if epochs_not_improving > args.early_stop:
            break
    
    # Output tracked statistics
    print('Final Arrays')
    print(train_loss_saved)
    print(train_r2_saved)
    print(val_loss)
    print(val_r2)

    print(f'Min Val Loss Epoch End {min_val_loss_epoch+1}')

