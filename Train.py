import datetime
import pickle

from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch
import argparse
from torchvision import transforms, datasets, models
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def array_random_pick(array, pick_num):
    index = np.arange(len(array))
    pick = np.random.choice(len(array), pick_num, replace=False)
    unpick = np.equal(np.in1d(index, pick), False)
    return array[unpick], array[pick]

def test_for_one_epoch(model, val_loader, device, loss_function):
    model.eval()
    with torch.no_grad():
        val_loss = []
        correct = 0
        total = 0
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data[0].to(device),
                val_data[1].to(device),
            )

            # if 'inception' in model_name:
            #     val_outputs, _ = model(val_inputs)
            # else:
            val_outputs = model(val_inputs)

            # compute metric for current iteration
            val_loss.append(loss_function(val_outputs, val_labels).item())

            _, pred = val_outputs.max(1)
            total += val_labels.size(0)
            correct += pred.eq(val_labels).sum().item()

        val_loss = torch.mean(torch.tensor(val_loss)).item()
        val_acc = correct / total
    return val_loss, val_acc

def train(max_epochs, model, loss_function, optimizer, val_interval,
          train_loader, device, val_loader, test_loader, model_name, train_ds, val_ds, root_dir, save_name):
    train_loss_lst = []
    train_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []
    highest_val_acc = -1
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        train_total = 0
        train_correct = 0
        for batch_data in train_loader:
            step += 1

            inputs, labels = (
                batch_data[0].to(device),
                batch_data[1].to(device),
            )

            optimizer.zero_grad()
            if 'inception' in model_name:
                logits, _ = model(inputs)
            else:
                logits = model(inputs)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()
            # epoch_loss += loss.item()
            epoch_loss += loss.item()

            _, train_pred = logits.max(1)
            train_total += labels.size(0)
            train_correct += train_pred.eq(labels).sum().item()


            # print(
            #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
            #     f"train_loss: {loss.item():.4f}")
        epoch_loss /= len(train_loader)
        epoch_acc = train_correct/train_total

        train_loss_lst.append(epoch_loss)
        train_acc_lst.append(epoch_acc)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")


        if (epoch + 1) % val_interval == 0:
            val_loss, val_acc = test_for_one_epoch(model, val_loader, device, loss_function)
            # model.eval()
            # with torch.no_grad():
            #     val_loss = []
            #     correct = 0
            #     total = 0
            #     for val_data in val_loader:
            #         val_inputs, val_labels = (
            #             val_data[0].to(device),
            #             val_data[1].to(device),
            #         )
            #
            #         # if 'inception' in model_name:
            #         #     val_outputs, _ = model(val_inputs)
            #         # else:
            #         val_outputs = model(val_inputs)
            #
            #         # compute metric for current iteration
            #         val_loss.append(loss_function(val_outputs, val_labels).item())
            #
            #         _, pred  = val_outputs.max(1)
            #         total += val_labels.size(0)
            #         correct += pred.eq(val_labels).sum().item()
            #
            #
            #     val_loss = torch.mean(torch.tensor(val_loss)).item()
            #     val_acc = correct / total

            print('val_loss', val_loss)
            print('val_acc', val_acc)
            val_acc_lst.append(val_acc)
            val_loss_lst.append(val_loss)

            if val_acc > highest_val_acc:
                torch.save(model.state_dict(), os.path.join(
                    root_dir, f"{save_name}-model.pth"))
                print(f"state saved at epoch {epoch}")
                highest_val_acc = val_acc





        # if (epoch + 1) % save_result_interval == 0:
        #
        #     with open(os.path.join(root_dir, "results.pickle"), 'wb') as f:
        #         pickle.dump([epoch_loss_values, dice_metric_values, metrics_values, RI_values, ARI_values], f)



    # print(
    #     f"train completed, best_metric: {best_metric:.4f} "
    #     f"at epoch: {best_metric_epoch}")
    # torch.save(model.state_dict(), os.path.join(
    #                     root_dir, f"{save_name}-model.pth"))



    # testing
    model.load_state_dict(torch.load(os.path.join(root_dir, f"{save_name}-model.pth")))
    test_loss, test_acc = test_for_one_epoch(model, test_loader, device, loss_function)
    test_acc_lst = [test_acc]
    test_loss_lst = [test_loss]
    # saving epoch_loss_values and metric_values
    with open(os.path.join(root_dir, f"{save_name}-results.pickle"), 'wb') as f:
        pickle.dump([train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst, test_loss_lst, test_acc_lst], f)
    return test_acc


def get_model(args):
    model_name = args['model']
    device = torch.device("cuda:0")
    if "inception" in model_name:
        model_name = "inceptionv3"
        if args['pretrain']:
            model = models.inception_v3(pretrained=True)
            model.fc = nn.Linear(2048, 12)
            model = model
        else:
            model = models.inception_v3(num_classes=12)

    elif "resnet" in model_name:
        model_name = "resnet18"
        if args['pretrain']:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 12)
            model = model
        else:
            model = models.resnet18(num_classes=12)
    return model_name, model

def main(args, root_dir, train_set, valid_set, test_set, test=False):
    # model_name = args['model']
    device = torch.device("cuda:0")
    # if "inception" in model_name:
    #     model_name = "inceptionv3"
    #     if args['pretrain']:
    #         model = models.inception_v3(pretrained=True)
    #         model.fc = nn.Linear(2048, 12)
    #         model = model.to(device)
    #     else:
    #         model = models.inception_v3(num_classes=12).to(device)
    #
    # elif "resnet" in model_name:
    #     model_name = "resnet18"
    #     if args['pretrain']:
    #         model = models.resnet18(pretrained=True)
    #         model.fc = nn.Linear(model.fc.in_features, 12)
    #         model = model.to(device)
    #     else:
    #         model = models.resnet18(num_classes=12).to(device)

    model_name, model = get_model(args)
    model = model.to(device)

    if not 0 <= args['data_size'] <= 1:
        print("need a valid data_size percentage: between 0 and 1")
        exit(1)

    save_name = args['save_name']
    # save_name = "savename"
    learning_rate = 0.001
    weight_decay = 1e-4


    batch_size = 32


    # whole_ds = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    #
    #
    # train_mask, valid_mask = array_random_pick(np.arange(len(whole_ds)), 500)
    #
    # train_set = torch.utils.data.Subset(whole_ds, train_mask)
    # valid_set = torch.utils.data.Subset(whole_ds, valid_mask)
    #
    # discard_mask, subset_mask = array_random_pick(np.arange(len(train_set)), int(round(args["data_size"] * len(train_set))))
    # train_set = torch.utils.data.Subset(train_set, subset_mask)

    print(len(train_set), f"images selected for training")
    print(len(valid_set), f"images selected for validation")
    print(len(test_set), f"images selected for testing")

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    val_interval = 1
    max_epochs = 40

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True)
    val_loader = DataLoader(valid_set,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=False)
    test_loader = DataLoader(test_set,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=False)
    if test:
        model.load_state_dict(torch.load(os.path.join('./2022-04-15-01-34', f"{save_name}-model.pth")))
        test_loss, test_acc = test_for_one_epoch(model, test_loader, device, loss_function)
        return test_acc
    else:
        test_acc = train(args['max_epoch'], model, loss_function, optimizer, val_interval, train_loader,
              device, val_loader, test_loader, model_name, train_set, valid_set, root_dir, save_name)
    return test_acc


if __name__ == '__main__':

    # USE_TERMINAL_ARGS = False
    # if USE_TERMINAL_ARGS:
    #     my_parser = argparse.ArgumentParser(description='Parameters')
    #
    #     my_parser.add_argument('-m', '--model', help='model type: inception or resnet', required=True)
    #     my_parser.add_argument('-p', '--pretrain', type=bool, help='pretrain model or not', required=True)
    #     my_parser.add_argument('-ds', '--data_size', type=float,
    #                            help='how many percent of the dataset is used, a float between 0 and 1', default=1)
    #     my_parser.add_argument('-n', '--save_name', help='the name of the file for saving model params and results', required=True)
    #     # Execute the parse_args() method
    #     args = vars(my_parser.parse_args())
    # else:
    #     args = {
    #         'model': 'resnet18',
    #         'pretrain': True,
    #         'data_size': 0.5,
    #         'save_name': 'testname'
    #     }
    data_path = './plant-seedlings-classification'
    root_dir = os.path.join('.', datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'))
    isExist = os.path.exists(root_dir)
    if not isExist: os.mkdir(root_dir)
    print('root_dir,', root_dir)

    transform = transforms.Compose([
        transforms.Resize([299, 299]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.3288, 0.2894, 0.2073), std=(0.1039, 0.1093, 0.1266))
    ])

    # data split
    # first, split into 0.2 test set, 0.8 other (we want to keep the test set consistent here)
    # second, take subset of the 0.8 other according to `data_size` in args
    # third, split the subset into 0.1 validation set (for taking the best model) and 0.9 training set

    whole_ds = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(data_path, 'mytest'), transform=transform)

    # # train_mask, valid_mask = array_random_pick(np.arange(len(whole_ds)), 500)
    # train_idx, test_idx = train_test_split(list(range(len(whole_ds.targets))), test_size=0.2, stratify=whole_ds.targets)
    #
    # whole_train_set = torch.utils.data.Subset(whole_ds, train_idx)
    # test_set = torch.utils.data.Subset(whole_ds, test_idx)
    #
    # whole_train_set_targets = [whole_train_set.dataset.targets[i] for i in whole_train_set.indices]
    # discard_mask, subset_mask = train_test_split(list(range(len(whole_train_set_targets))), test_size=0.2, stratify=whole_train_set_targets)
    # subset_train_set = torch.utils.data.Subset(whole_train_set, subset_mask)
    #
    # subset_train_set_targets = [subset_train_set.dataset.dataset.targets[i] for i in subset_train_set.indices]
    # train_idx, val_idx = train_test_split(list(range(len(subset_train_set_targets))), test_size=0.1, stratify=subset_train_set_targets)
    # train_set = torch.utils.data.Subset(subset_train_set, train_idx)
    # val_set = torch.utils.data.Subset(subset_train_set, val_idx)


    model_name_lst = ['resnet18', 'inceptionv3']
    ds_size_lst = [0.1, 0.25, 0.5, 0.75, 1]
    pretrain_lst = [True, False]

    result_lst = []
    for model in model_name_lst:
        for ds_size in ds_size_lst:
            for pretrain in pretrain_lst:
                save_name = f"{model}-{str(ds_size)}-{str(pretrain)}-80"
                args = {
                        'model': model,
                        'pretrain': pretrain,
                        'data_size': ds_size,
                        'save_name': save_name,
                        'max_epoch': 80
                }
                if ds_size < 0.6:
                    args['max_epoch'] = 40

                if ds_size == 1:
                    subset_train_set = torch.utils.data.Subset(whole_ds, np.arange(len(whole_ds)))
                else:
                    discard_mask, subset_mask = train_test_split(list(range(len(whole_ds.targets))), test_size=ds_size, stratify=whole_ds.targets)
                    subset_train_set = torch.utils.data.Subset(whole_ds, subset_mask)

                subset_train_set_targets = [subset_train_set.dataset.targets[i] for i in subset_train_set.indices]
                train_idx, val_idx = train_test_split(list(range(len(subset_train_set_targets))), test_size=0.1, stratify=subset_train_set_targets)
                train_set = torch.utils.data.Subset(subset_train_set, train_idx)
                val_set = torch.utils.data.Subset(subset_train_set, val_idx)

                test_acc = main(args, root_dir, train_set, val_set, test_set)
                args['test_acc'] = test_acc
                result_lst.append(args)


    print(result_lst)




    # model_name = args['model']
    # size = 299
    # device = torch.device("cuda:0")
    # if "inception" in model_name:
    #     model_name = "inceptionv3"
    #     if args['pretrain']:
    #         model = models.inception_v3(pretrained=True)
    #         model.fc = nn.Linear(2048, 12)
    #         model = model.to(device)
    #     else:
    #         model = models.inception_v3(num_classes=12).to(device)
    #
    # elif "resnet" in model_name:
    #     model_name = "resnet18"
    #     if args['pretrain']:
    #         model = models.resnet18(pretrained=True)
    #         model.fc = nn.Linear(model.fc.in_features, 12)
    #         model = model.to(device)
    #     else:
    #         model = models.resnet18(num_classes=12).to(device)
    #
    # if not 0 <= args['data_size'] <= 1:
    #     print("need a valid data_size percentage: between 0 and 1")
    #     exit(1)
    #
    # save_name = args['save_name']
    # # save_name = "savename"
    # learning_rate = 0.001
    # weight_decay = 1e-4
    # data_path = './plant-seedlings-classification'
    # root_dir = os.path.join('.', datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'))
    # isExist = os.path.exists(root_dir)
    # if not isExist: os.mkdir(root_dir)
    # print('root_dir,', root_dir)
    #
    # batch_size = 32
    #
    # transform = transforms.Compose([
    #     transforms.Resize([size, size]),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=(0.3288, 0.2894, 0.2073), std=(0.1039, 0.1093, 0.1266))
    # ])
    # whole_ds = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    #
    # train_mask, valid_mask = array_random_pick(np.arange(len(whole_ds)), 500)
    #
    # train_set = torch.utils.data.Subset(whole_ds, train_mask)
    # valid_set = torch.utils.data.Subset(whole_ds, valid_mask)
    #
    # discard_mask, subset_mask = array_random_pick(np.arange(len(train_set)), int(round(args["data_size"] * len(train_set))))
    # train_set = torch.utils.data.Subset(train_set, subset_mask)
    #
    # print(len(train_set), f"images selected for training")
    # print(len(valid_set), f"images selected for testing")
    #
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # val_interval = 1
    # max_epochs = 40
    #
    # train_loader = DataLoader(train_set,
    #                           batch_size=batch_size,
    #                           num_workers=0,
    #                           shuffle=True)
    # val_loader = DataLoader(valid_set,
    #                         batch_size=batch_size,
    #                         num_workers=0,
    #                         shuffle=False)
    # train(max_epochs, model, loss_function, optimizer, val_interval, train_loader,
    #       device, val_loader, model_name, train_set, valid_set, root_dir, save_name)




    # print(len(train_set),len(valid_set))

    # print(valid_set[401])
    # loader = DataLoader(train_ds,
    #                          batch_size=10,
    #                          num_workers=0,
    #                          shuffle=False)
    #
    # mean = 0.0
    # for images, _ in loader:
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    # mean = mean / len(loader.dataset)
    #
    # var = 0.0
    # for images, _ in loader:
    #     batch_samples = images.size(0)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    # std = torch.sqrt(var / (len(loader.dataset)*224*224))
    #
    # print(mean)
    # print(std)
