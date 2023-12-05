import torch
import numpy as np
import time

def test_model(model, test_loader, platform):
    
    device = torch.device(platform)
    
    path_list = []
    output_list = []
    label_list = []
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        num_correct_neutral = 0
        num_correct_selection = 0
        num_total_neutral = 0
        num_total_selection = 0
        
        start_time = time.perf_counter()
        for i, data in enumerate(test_loader):
            paths, inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)          
            outputs = model(inputs)

            num_correct_neutral += torch.sum((torch.argmax(outputs, dim=1) == labels) & (labels == 0))
            num_correct_selection += torch.sum((torch.argmax(outputs, dim=1) == labels) & (labels == 1))
            
            num_total_neutral += torch.sum(labels == 0).item()
            num_total_selection += torch.sum(labels == 1).item()
            output_list.append(outputs)
            label_list.append(labels)
            path_list += list(paths)
            
        print("Neutral Acc: ", num_correct_neutral.item()/num_total_neutral)
        print("Selection Acc: ", num_correct_selection.item() / num_total_selection)
        print("Total Acc: ", (num_correct_selection.item() + num_correct_neutral.item()) / (num_total_neutral + num_total_selection))
        
        end_time = time.perf_counter()
        inference_time = end_time-start_time
        output_list = np.array(torch.concatenate(output_list, axis=0).cpu())
        label_list = np.array(torch.concatenate(label_list, axis=0).cpu())
    return path_list, output_list, label_list, inference_time