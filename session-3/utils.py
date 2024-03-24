import torch


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc
    

def binary_accuracy(labels, outputs):
    preds = outputs.round()
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc


def binary_accuracy_with_logits(labels, outputs):
    preds = torch.sigmoid(outputs).round()  # Apply sigmoid and round to get binary predictions
    correct = torch.eq(preds.squeeze(), labels.float()).float()  # Ensure labels are float for comparison
    acc = correct.mean()
    return acc.item()


def save_model(model, path):
    torch.save(model.state_dict(), path)
