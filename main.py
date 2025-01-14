from Logic.dataLoad import get_loader
from Logic.models import SweepNet
from Logic.testing import test_model
from Logic.training import train_model
import torch
import numpy as np
import os
import getopt
import time
import sys
    
def train(height, width, epochs, batch, platform, opath, ipath, model_class):
    print("Training model ", opath)
       
    lr = 0.5e-3
    channels = 1

    if model_class == "SweepNet":
        model = SweepNet(height, width, channels=channels)
    else:
        raise ValueError("Unknown model class")
        
    validation = True

    train_loader, val_loader = get_loader(ipath, 128, class_folders=True, shuffle=True, shuffle_row=True, mix_images=False, validation=validation)
    model, history = train_model(model, train_loader, epochs, lr=lr, loss_weights=None, platform=platform, val_loader=val_loader, mini_batch_size=batch)
    torch.save(model.state_dict(), os.path.join(opath, 'model.pt'))
    
        
def test(height, width, platform, mpath, ipath, opath, model_class):
    print("Testing model ", mpath)
    
    # set class folders to false for production
    test_loader, _ = get_loader(ipath, batch_size=128, class_folders=True, shuffle=False, shuffle_row=False, mix_images=False, validation=False)
              
    channels = 1
    
    if model_class == "SweepNet":
        model = SweepNet(height, width, channels=channels)
    else:
        raise ValueError("Unknown model class")
    
    init_state = torch.load(os.path.join(mpath, 'model.pt'), map_location=torch.device(platform))
    model.load_state_dict(init_state)
    
    resultsData = np.empty((0, 4), float)
    
    start = time.perf_counter()
    path_list, output_list, label_list, inference_time = test_model(model, test_loader, platform=platform)
    print(time.perf_counter() - start)

    probability_tensor = torch.nn.functional.softmax(torch.Tensor(output_list), dim=1)[:, 1]
    for path, probability, label in zip(path_list, probability_tensor, label_list):
        path = os.path.split(path)[1]
        resultsData = np.append(resultsData,
                                np.array([[path,
                                        label,
                                        float(1-probability),
                                        float(probability)
                                        ]]),
                                axis=0)
    
    resultsData = resultsData[resultsData[:, 0].argsort()]
    np.savetxt(os.path.join(opath, 'PredResults.txt'), resultsData[:][:], fmt="%s")


        

# a parameter to select which model to use is required to be implemented.
def main(argv):

    opts, args = getopt.getopt(argv, "m:p:t:e:i:o:d:h:w:c:f:x:y:b:H", ["mode=", "platform=", "threads=", "epochs=", "ipath=", "opath=", "modeldirect=", "height=", "width=", "class=", "file=", "distance=", "detect=", "batch=", "help="])
 
    for opt, arg in opts:
        print(opt, arg)
        if opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-p", "--platform"):
            platform = arg
        elif opt in ("-t", "--threads"):
            threads = arg
        elif opt in ("-e", "--epochs"):
            epochs = arg
        elif opt in ("-i", "--ipath"):
            ipath = arg
        elif opt in ("-o", "--opath"):
            opath = arg
        elif opt in ("-d", "--modeldirect"):
            mpath = arg
        elif opt in ("-h", "--height"):
            height = arg
        elif opt in ("-w", "--width"):
            width = arg
        elif opt in ("-c", "--class"):
            model_class = arg
        elif opt in ("-f", "--file"):
            load_binary = arg
        elif opt in ("-x", "--distance"):
            use_bp_distance = arg
        elif opt in ("-y", "--detect"):
            train_detect = arg
        elif opt in ("-b", "--batch"):
            batch = arg
        elif opt in ("-H", "--help"):
            help()
            return 0
	
    if not os.path.exists(opath):
        os.makedirs(opath)
        
    torch.set_num_threads(int(threads))

    if (mode == "train"):
        start=time.time()
        train(int(height), int(width), int(epochs), int(batch), platform, opath, ipath, model_class)
        end=time.time()
        with open(opath + "/image-dimensions.txt", "w") as f:
            f.write(str(str(height) + " " + str(width)))
            
            
    elif (mode == "predict"):
        start=time.time()
        test(int(height), int(width), platform, mpath, ipath, opath, model_class)
        end=time.time()
        
    else:
        print("No valid mode detected")
        return 0
			
if __name__ == "__main__":
    main(sys.argv[1:])		
