from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import pdb
def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(iterable=train_loader):
            optimizer.zero_grad()

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(output, target.type_as(output))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Training loss is {train_loss/len(train_loader)}")
        criterion_mse = nn.MSELoss().to(device)
        val_loss_mse = evaluate(model=model, criterion=criterion_mse, dataloader=val_loader, device=device)
        val_score_pearson = evaluate(model=model, criterion='pearson', dataloader=val_loader, device=device)
        print("Epoch {} complete! Validation Loss MSE: {}".format(epoch+1, val_loss_mse))
        print("Epoch {} complete! Validation Score pearson: {}".format(epoch+1, val_score_pearson))

def evaluate(model, criterion, dataloader, device):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
            if criterion == 'pearson':
                mean_loss += pearson(output, target.type_as(output)).item()
            elif criterion == 'spearman':
                mean_loss += spearman_correlation(torch.Tensor(output.to("cpu")), torch.Tensor(target.type_as(output).to("cpu")))
            else:
                mean_loss += criterion(output, target.type_as(output)).item()
#             mean_err += get_rmse(output, target)
            count += 1

    return mean_loss/count

def get_rmse(output, target):
    err = torch.sqrt(metrics.mean_squared_error(target, output))
    return err

def predict(model, dataloader, device):
    predicted_label = []
    actual_label = []
    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):

            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)

            predicted_label += output
            actual_label += target

    return predicted_label

def pearson(output,target):
    x = output
    y = target
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1)
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    # v = torch.arange(len(x)).reshape(-1, 1, 1)
    ranks[tmp] = torch.arange(len(x))
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    # pdb.set_trace()
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)