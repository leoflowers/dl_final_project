import torch

def train(model, dataloader, criterion, optimizer, epochs, device):
    size = len(dataloader.dataset) 
   
    print(f"Using device '{device}' for training")
    model.model.train()

    for i in range(epochs):
        print(f"---------------------- epoch {i+1}/{epochs} ----------------------")

        epoch_loss, epoch_acc = 0.0, 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
        
            with torch.set_grad_enabled(True):
                inputs = inputs.to(device).requires_grad_()
                labels = labels.to(device)

                predictions = model.model(inputs)
                loss = criterion(predictions, labels)

                loss.backward()
                optimizer.step()
        
            if idx%100 == 0:
                loss, currentIdx = loss.item(), idx*len(inputs)
                print(f"loss: {loss:>7f} [{currentIdx:>5d}/{size:5>d}]")
