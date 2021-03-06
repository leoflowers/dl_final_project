import torch

def train(model, dataloader, criterion, optimizer, device, distributed):
    size = len(dataloader.dataset) 
   
    epoch_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
    
        with torch.set_grad_enabled(True):
            inputs = inputs.to(device).requires_grad_()
            labels = labels.to(device)

            if distributed:
                predictions = model(inputs)
            else:
                predictions = model.model(inputs)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()
       
            epoch_loss += loss.item()

        if idx%100 == 0:
            loss, currentIdx = loss.item(), idx*len(inputs)
            print(f"loss: {loss:>7f} [{currentIdx:>5d}/{size:5>d}]")


    return epoch_loss / len(dataloader)
