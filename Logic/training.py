import torch
import numpy as np
import time
from Logic.testing import test_model


def train_model(model, train_loader, epochs, lr, platform, loss_weights=None, val_loader=None, mini_batch_size=None):
    
    device = torch.device(platform)
    
    model.train(True)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights) # torch.tensor(loss_weights, device=platform))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    batch_accuracies = []
    val_accuracies = []
    batch_losses = []
    prev_val_acc = 0
    
    if mini_batch_size == None:
        mini_batch_size = train_loader.batch_size
    
    model.to(device)
    start_time = time.perf_counter()
    for epoch in range(epochs):
        running_loss = 0
        batch_correct = 0
        batch_total = 0
        
        for i, super_batch in enumerate(train_loader):
            super_paths, super_inputs, super_labels = super_batch
            super_inputs, super_labels = super_inputs.to(device), super_labels.to(device) 
            for inputs, labels in zip(torch.split(super_inputs, mini_batch_size, dim=0), torch.split(super_labels, mini_batch_size, dim=0)):
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # if labels is probability over classes, correct is label closest to classes
                if len(labels.shape) == 1:
                    batch_correct += torch.sum((torch.argmax(outputs, dim=1) == labels))
                else:
                    batch_correct += torch.sum((torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)))
                    
                batch_total += labels.shape[0]
                
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            
        batch_accuracy = batch_correct / batch_total
        batch_loss = running_loss / len(train_loader)
        batch_accuracies.append(batch_accuracy)
        batch_losses.append(batch_loss)
        
        if val_loader != None:
            _, val_outputs, val_labels, _ = test_model(model, val_loader, platform=platform)
            model.train(True)
            
            # if labels is probability over classes, correct is label closest to classes
            if len(val_labels.shape) == 1:
                num_correct = np.sum((np.argmax(val_outputs, axis=1) == val_labels))
            else:
                num_correct = np.sum((np.argmax(val_outputs, axis=1) == np.argmax(val_labels, axis=1)))
            
            num_total = val_labels.shape[0]
            val_accuracy = num_correct / num_total
            val_accuracies.append(val_accuracy)
            print("epoch {}, loss {}, acc {}, val_acc {}".format(epoch, batch_loss, batch_accuracy, val_accuracy))
            if val_accuracy >= prev_val_acc:
                print("Saving best model")
                prev_val_acc = val_accuracy
                best_state = {}
                for key in model.state_dict():
                    best_state[key] = model.state_dict()[key].clone()
        else:         
            print("epoch {}, loss {}, acc {}".format(epoch, batch_loss, batch_accuracy))
        
    end_time = time.perf_counter()
    inference_time = end_time-start_time
    batch_accuracies = np.array(torch.Tensor(batch_accuracies).cpu())
    batch_losses = np.array(torch.Tensor(batch_losses).cpu())
    val_accuracies = np.array(val_accuracies)
    
    if val_loader != None:
        model.load_state_dict(best_state)
        
    return model, (batch_accuracies, batch_losses, inference_time, val_accuracies)
