import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, MyNet
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]

    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'mynet':
        model = MyNet()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    train_loss_list = []
    train_accu_list = []
    valid_loss_list = []
    valid_accu_list = []

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params of "+model_type+":"+str(pytorch_total_params))
    pytorch_trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params of "+model_type+":"+str(pytorch_trainable_total_params))

    # Run any number of epochs you want
    ep = 10
    for epoch in range(1,ep+1):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))
                if batch == len(train_loader):
                    train_loss_list.append(ave_loss)
                    train_accu_list.append(acc)

        ################
        ## Validation ##
        ################
        model.eval()

        val_correct_cnt, val_total_loss, val_total_cnt = 0, 0, 0
        best_accu = 0
        with torch.no_grad():
            for batch, (val_x, val_label) in enumerate(val_loader,1):
                if use_cuda:
                    val_x, val_label = val_x.cuda(), val_label.cuda()
                val_out = model(val_x)

                val_loss = criterion(val_out, val_label)

                val_total_loss += val_loss.item()
                _, val_pred_label = torch.max(val_out, 1)
                val_total_cnt += val_x.size(0)
                val_correct_cnt += (val_pred_label == val_label).sum().item()
        
                if batch == len(val_loader):
                    val_acc = val_correct_cnt / val_total_cnt
                    val_ave_loss = val_total_loss / batch
                    print ('Validation batch index: {}, validation loss: {:.6f}, acc: {:.3f}'.format(
                        batch, val_ave_loss, val_acc))
                    valid_loss_list.append(val_ave_loss)
                    valid_accu_list.append(val_acc)
                    if val_acc>best_accu:
                        best_accu = val_acc
                        if best_accu>0.985:
                            torch.save(model.state_dict(), './checkpoint/%s.pth' % (model.name()+'_epc'+str(epoch)+'_accu'+str(int(best_accu*1000))))

        
        model.train()

    # Save trained model
    #torch.save(model.state_dict(), './checkpoint/%s.pth' % (model.name()+'_epc'+str(ep)+'_accu'+str(int(valid_accu_list[-1]*1000))))

    # Plot Learning Curve
    x_tick = list(range(ep))
    fig = plt.figure()
    plt.plot(x_tick, train_loss_list,'-',label='Training Loss')
    plt.title("Training Loss_"+model_type)
    plt.xlabel("epoches")
    plt.ylabel("Loss")
    plt.legend()

    fig = plt.figure()
    plt.plot(x_tick, train_accu_list,'-',label='Training Accuracy')
    plt.title("Training Accuracy_"+model_type)
    plt.xlabel("epoches")
    plt.ylabel("Accuracy")
    plt.legend()

    fig = plt.figure()
    plt.plot(x_tick, valid_loss_list,'-',label='Validation Loss')
    plt.title("Validation Loss_"+model_type)
    plt.xlabel("epoches")
    plt.ylabel("Loss")
    plt.legend()

    fig = plt.figure()
    plt.plot(x_tick, valid_accu_list,'-',label='Validation Accuracy')
    plt.title("Validation Accuracy_"+model_type)
    plt.xlabel("epoches")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    