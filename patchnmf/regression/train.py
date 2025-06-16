from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from patchnmf.regression.models import LinearRegression
from patchnmf.regression.plot_regression import plot_lambda_optimisation

def train_model(model, X, y, n_epochs=100, lr=0.001, weight_decay=0.0001, device='cpu', batch_size=100):
    
    n_batches = int(X.shape[1] / batch_size)

    criterion = nn.MSELoss()

    # weight decay is the same as L2 regularization but only for SGD (not Adam or other optimisers!!!)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_loss = []
    train_acc = []
    
    for epoch in range(n_epochs):
        
        model.train()

        running_loss = 0.0
        
        for i in range(n_batches):
            data = X[:, i*batch_size:(i+1)*batch_size].T
            target = y[i*batch_size:(i+1)*batch_size]

            data, target = data.to(device), target.to(device)
            output = model(data)
            optimizer.zero_grad()
            
            # print shapes
            # print(f'data.shape: {data.shape}')
            # print(f'target.shape: {target.shape}')
            # print(f'output.shape: {output.shape}')
            # print(f'target[:, None].shape: {target[:, None].shape}')
            # print(f'target[:, None].shape: {target[:, None].squeeze().shape}')

            

            loss = criterion(output, target[:, None]) if len(target.shape) == 1 else criterion(output, target)

            # print(f'loss: {loss}')

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    
        train_loss.append(running_loss)

    return train_loss


def eval_model(model, X, y, device, title='', plot=False):

    #loss = nn.MSELoss() some overwriting error (loss function owewritten with loss value)
    criterion = torch.nn.MSELoss() #this added instead
    loss_value = criterion(y_pred, y)

    model.eval()
    with torch.no_grad():
        X = (X.T).clone().float().to(device)
        y = (y).clone().float().to(device)
        y_pred = model(X).squeeze() # IMPORTANT: squeeze to remove singleton dimension (otherwise it will be broadcasted and will give wrong results)
        loss = loss(y_pred, y)
        y_pred = y_pred.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        mse = np.mean((y_pred - y)**2)

    return loss_value.item(), y_pred


def optimise_weight_decay(X_train, y_train, X_val, y_val, all_weight_decay=None, n_epochs=1000, lr=0.001, batch_size=225, plot_fname_path=None):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_train_loss_all_t = []
    all_train_loss = []
    all_val_loss = []
    all_train_pred = []
    all_val_pred = []

    for weight_decay in tqdm(all_weight_decay):
        # if y is a 1-d vector then output_size=1, otherwise output_size=y_train.shape[1]
        output_size = 1 if len(y_train.shape) == 1 else y_train.shape[1]

        model = LinearRegression(input_size=X_train.shape[0], output_size=output_size)

        model = model.to(device)

        train_loss_all_t = train_model(model, X_train, y_train, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay, device=device, batch_size=batch_size)

        _, train_pred = eval_model(model, X_train, y_train, device=device, title='train')
        val_loss, val_pred = eval_model(model, X_val, y_val, device=device, title='val')

        train_loss = np.min(train_loss_all_t)
        
        all_train_loss_all_t.append(train_loss_all_t)
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_pred.append(train_pred)
        all_val_pred.append(val_pred)

    # now choose best weight decay, refit and visualise results
    best_idx = np.argmin(all_val_loss)
    best_weight_decay = all_weight_decay[best_idx]

    model = LinearRegression(input_size=X_train.shape[0], output_size=output_size)
    model = model.to(device)

    train_loss_all_t = train_model(model, X_train, y_train, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay, device=device, batch_size=batch_size)

    _, train_pred = eval_model(model, X_train, y_train, device=device, title='train')
    val_loss, val_pred = eval_model(model, X_val, y_val, device=device, title='val')

    train_loss = np.min(train_loss_all_t)


    # plot results
    plot_lambda_optimisation(all_weight_decay, all_train_loss, all_val_loss, best_weight_decay, best_idx, all_train_pred, all_val_pred, y_train, y_val, all_train_loss_all_t, plot_fname_path=plot_fname_path)

    return best_weight_decay