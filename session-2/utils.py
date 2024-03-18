import torch


def accuracy(outputs, labels):
    # Assuming outputs are raw logits, you'd first apply softmax (if needed) and then argmax to get predicted class indices
    preds = outputs.argmax(dim=1)  # Get the index of the max log-probability
    correct = preds.eq(labels).sum()  # Element-wise equality, summed up for total correct predictions
    acc = correct.float() / labels.size(0)  # Calculate accuracy
    return acc.item()  # Return as Python float

    

def save_model(model, path):
    torch.save(model.state_dict(), path)