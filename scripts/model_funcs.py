import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from livelossplot import PlotLosses


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return True


def mean_f1_score(true_pos, false_pos, false_neg):
    """
    Compute the mean F1-score from arrays of true positives, false positives
    and false negatives. 
    """
    all_pred_pos = true_pos + false_pos
    all_act_pos = true_pos + false_neg

    prec = np.divide(true_pos, all_pred_pos, where=(all_pred_pos != 0))
    rec = np.divide(true_pos, all_act_pos, where=(all_act_pos != 0))
    f1 = np.divide(2*prec*rec, prec+rec, where=(prec+rec != 0))

    return np.mean(f1)


def train(model, optimizer, criterion, data_loader, device):
    model.train()
    train_loss = 0.
    mcm = np.zeros((200, 2, 2))
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        a2 = model(X)
        loss = criterion(a2, y)
        loss.backward()
        train_loss += loss*X.size(0)
        y_pred = F.log_softmax(a2, dim=1).max(1)[1]
        mcm += multilabel_confusion_matrix(y.cpu().numpy(),
                                           y_pred.detach().cpu().numpy(), labels=list(range(200)))
        optimizer.step()

    return train_loss/len(data_loader.dataset), mean_f1_score(mcm[:, 1, 1], mcm[:, 0, 1], mcm[:, 1, 0])


def validate(model, criterion, data_loader, device):
    model.eval()
    validation_loss = 0.
    mcm = np.zeros((200, 2, 2))
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X)
            loss = criterion(a2, y)
            validation_loss += loss*X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            mcm += multilabel_confusion_matrix(y.cpu().numpy(),
                                               y_pred.detach().cpu().numpy(), labels=list(range(200)))

    return validation_loss/len(data_loader.dataset), mean_f1_score(mcm[:, 1, 1], mcm[:, 0, 1], mcm[:, 1, 0])


def evaluate(model, data_loader, device):
    model.eval()
    y_preds = []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            y_preds.append(y_pred.cpu().numpy())

    return np.concatenate(y_preds, 0)


def train_model(model, device, train_loader, validation_loader, n_epochs, lr, momentum, weight_decay, save_path=None, seed=None):
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    min_loss = float("inf")
    liveloss = PlotLosses()
    for epoch in range(n_epochs):
        logs = {}
        train_loss, train_accuracy = train(model, optimizer,
                                           criterion, train_loader, device)

        logs['' + 'log loss'] = train_loss.item()
        logs['' + 'accuracy'] = train_accuracy.item()

        validation_loss, validation_accuracy = validate(model, criterion,
                                                        validation_loader, device)
        logs['val_' + 'log loss'] = validation_loss.item()
        logs['val_' + 'accuracy'] = validation_accuracy.item()

        liveloss.update(logs)
        liveloss.draw()

        # Store best validation loss model
        if save_path is not None and validation_loss < min_loss:
            torch.save(model.state_dict(),
                       save_path+F"model_state_"+str(epoch)+".pth")
            min_loss = validation_loss

    return model
