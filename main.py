import sys

import os
import json
import time
import utils
import models.DeepTTE
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np
from itertools import repeat

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str, default = 'train')
parser.add_argument('--batch_size', type = int, default = 8)
parser.add_argument('--epochs', type = int, default = 10)

# evaluation args
parser.add_argument('--weight_file', type = str, default = './saved_weights/log_2021_05_18_11_50_45_193746')
parser.add_argument('--result_path', type = str, default = './result/')

# cnn args
parser.add_argument('--kernel_size', type = int, default = 3)

# rnn args
parser.add_argument('--pooling_method', type = str, default = 'attention')

# multi-task args
parser.add_argument('--alpha', type = float, default = 0.3)

# log file name
parser.add_argument('--log_file', type = str, default = 'log')

args = parser.parse_args()

config = json.load(open('./config_2.json', 'r'))

def train(model, elogger, train_set, eval_set):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    if torch.cuda.is_available():
        model.cuda()
    model.cpu()

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)


    for epoch in range(args.epochs):
        print('Training on epoch {}'.format(epoch))
        for input_file in train_set:
            print('Train on file {}'.format(input_file))

            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader_bewerkt.get_loader(input_file, args.batch_size)
            
            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                _, _, _, loss = model.eval_on_batch(attr, traj, config)

                # update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.data
                print('\r Progress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0))),
            print
            elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss / (idx + 1.0)))

        # evaluate the model after each epoch
        evaluate(model, elogger, eval_set, save_result = False)
        model.train()
        
        # save the weight file after each epoch
        weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")))
        elogger.log('Save weight file {}'.format(weight_name))
        torch.save(model.state_dict(), './saved_weights/' + weight_name)

def write_result(fs_entire, fs_local, pred_dict, pred_dict_local, local_length, attr, traj, traj_sectie):
    pred = pred_dict['pred'].data.numpy()
    label = pred_dict['label'].data.cpu().numpy()
    pred_local = pred_dict_local['pred'].data.cpu().numpy()
    label_local = pred_dict_local['label'].data.cpu().numpy()

    list_full_seq_sectie = []
    list_full_seq_coord_begin = []
    list_full_seq_coord_end = []
    for i in range(0,len(traj_sectie)):
        first_seq_sectie = traj_sectie[i][:-args.kernel_size + 1]
        second_seq_sectie = traj_sectie[i][args.kernel_size - 1:]
        first_seq_lats = traj['lats_original'][i][:-args.kernel_size + 1]
        second_seq_lats = traj['lats_original'][i][args.kernel_size - 1:]
        first_seq_lngs = traj['lngs_original'][i][:-args.kernel_size + 1]
        second_seq_lngs = traj['lngs_original'][i][args.kernel_size - 1:]
        full_seq_sectie = [str(x) + "--" + str(y) for x,y in zip(first_seq_sectie, second_seq_sectie)]
        full_seq_coord_begin = [str(x) + " " + str(y) for x,y in zip(first_seq_lats, first_seq_lngs)]
        full_seq_coord_end = [str(x) + " " + str(y) for x,y in zip(second_seq_lats, second_seq_lngs)]
        list_full_seq_sectie.append(full_seq_sectie)
        list_full_seq_coord_begin.append(full_seq_coord_begin)
        list_full_seq_coord_end.append(full_seq_coord_end)
    flat_list_full_seq_sectie = [item for sublist in list_full_seq_sectie for item in sublist]
    flat_list_full_seq_coord_begin = [item for sublist in list_full_seq_coord_begin for item in sublist]
    flat_list_full_seq_coord_end = [item for sublist in list_full_seq_coord_end for item in sublist]

    flat_list_full_seq_sectie = np.array(flat_list_full_seq_sectie).reshape(-1, 1)
    flat_list_full_seq_coord_begin = np.array(flat_list_full_seq_coord_begin).reshape(-1, 1)
    flat_list_full_seq_coord_end = np.array(flat_list_full_seq_coord_end).reshape(-1, 1)
    
    local_driverID_list = []
    local_dateID_list = []
    local_timeID_list = []
    local_treinID_list = []

    for i in range(pred_dict['pred'].size()[0]):
        dateID = attr['dateID'].data.cpu().numpy()[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data.cpu().numpy()[i]
        treinID = attr['Treinnr'].data.cpu().numpy()[i]
        dist = attr['dist'].data.cpu().numpy()[i]
        
        local_driverID_list.extend(repeat(driverID, local_length[i]))
        local_dateID_list.extend(repeat(dateID, local_length[i]))
        local_timeID_list.extend(repeat(timeID, local_length[i]))
        local_treinID_list.extend(repeat(treinID, local_length[i]))
        fs_entire.write('%.0f %.0f %.0f %.0f %.6f %.6f\n' % (dateID, timeID, driverID, treinID, label[i][0], pred[i][0]))
        
    for j in range(pred_dict_local['pred'].size()[0]):
        fs_local.write('%.0f %.0f %.0f %.0f %s %s %s %.6f %.6f\n' % (local_dateID_list[j], local_timeID_list[j], local_driverID_list[j], local_treinID_list[j], flat_list_full_seq_sectie[j][0], flat_list_full_seq_coord_begin[j][0], flat_list_full_seq_coord_end[j][0], label_local[j][0], pred_local[j][0]))
   
def evaluate(model, elogger, files, save_result = False):
    model.eval()
    if save_result:
        fs_entire = open('%s' % args.result_path + '/entire.res', 'w')
        fs_local = open('%s' % args.result_path + '/local.res', 'w')
        fs_entire.write('dateID timeID trainType TreinID label pred\n')
        fs_local.write('dateID timeID trainType TreinID Sectie lat_begin lng_begin lat_end lng_end label pred\n')
        
    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader_bewerkt.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            traj_sectie = traj['Sectie']
            
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            pred_dict, pred_dict_local, local_length, loss = model.eval_on_batch(attr, traj, config)

            if save_result: write_result(fs_entire, fs_local, pred_dict, pred_dict_local, local_length, attr, traj, traj_sectie)

            running_loss += loss.data

        print('Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx + 1.0)))
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))

    if save_result: 
        fs_entire.close()
        fs_local.close()

def get_kwargs(model_class):
    model_args = inspect.getargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs

def run():
    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)

    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result = True)

if __name__ == '__main__':
    run()