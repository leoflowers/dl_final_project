import torch


def test(model, dataloader, criterion, device, distributed):
    test_loss, correct = 0.0, 0
    
    #model.model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if distributed:
                predictions = model(inputs)
            else:
                predictions = model.model(inputs)
            loss = criterion(predictions, labels)
            
            test_loss += loss.item()
            predictions = torch.argmax(predictions, dim=1, keepdim=True).flatten()
            correct += (predictions == labels).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    
    epoch_acc = correct*100
    #print(f"Test error:\n\tAccuracy: {(100*correct):>0.1f}%, average loss: {test_loss:>8f}\n")
    return epoch_acc, test_loss
