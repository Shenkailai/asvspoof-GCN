import argparse
import sys
import os

import librosa
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from model import RawNet
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
import logging
from graph.graphcnn import Graph_CNN_ortega
from torchvision import transforms
import eval_metrics as em
from util import load_data

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"
__credits__ = ["Jose Patino", "Massimiliano Todisco", "Jee-weon Jung"]


def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    key_list = []
    score_list = []
    for batch_x, batch_y, batch_meta in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        # batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        batch_score = (batch_out[:, 1] - batch_out[:, 0]
                       ).data.cpu().numpy().ravel()
        key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        score_list.extend(batch_score.tolist())
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total), np.array(key_list), np.array(score_list)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()

    for batch_x, utt_id in data_loader:
        fname_list = []
        score_list = []
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]
        ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()
    print('Scores saved to {}'.format(save_path))


def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y, batch_meta in train_loader:

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        # batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct / num_total) * 100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len) + 1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


def compute_mfcc_feats(x):
    # mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24)
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=35)
    # delta = librosa.feature.delta(mfcc)
    # delta2 = librosa.feature.dela(delta)
    # feats = np.concatenate((mfcc, delta, delta2), axis=0)
    feats = np.swapaxes(mfcc, 0, 1)
    return feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default=r'F:\Dataset\ASVspoof2021-DF\data/ASVspoof_database/DF/',
                        help='Change this to user\'s full directory address of DF database. We assume that all three '
                             'ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same '
                             'database_path directory.')
    '''
    % database_path/
    %   |- DF
    %      |- ASVspoof2021_DF_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str,
                        default=r'F:\Dataset\ASVspoof2021-DF\protocols/ASVspoof_database/',
                        help='Change with path to user\'s DF ASVspoof2021 database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_DF_cm_protocols
    %      |- ASVspoof2021.DF.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt 
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')

    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='Model checkpoint. Here provide LA trained model path to evaluate on DF Eval databse')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF', choices=['LA', 'PA', 'DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False, help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True,
                        help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False,
                        help='use cudnn-benchmark? (default false)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling over nodes in a graph to get graph embeddig: sum or average')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of layers INCLUDING the input one (default: 2)')
    parser.add_argument('--features', type=str, default='mfcc')

    parser.add_argument('--graph_type', type=str, default="line", choices=["line", "cycle"],
                        help='Graph construction options')
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    # make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track

    assert track in ['LA', 'PA', 'DF'], 'Invalid track given'

    # Database
    prefix = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)

    # define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr, args.features)
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join("./logs", model_tag + ".log"), level=logging.DEBUG, format=LOG_FORMAT)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    if args.features == 'mfcc':
        feature_fn = compute_mfcc_feats

    transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: librosa.util.normalize(x),
        lambda x: feature_fn(x),
        lambda x: np.array(x)
    ])

    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path + '{}_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'.format(prefix)),
        is_train=True, is_eval=False)
    print('no. of training trials', len(file_train))

    # train_set=Dataset_ASVspoof2019_train(list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(
    # args.database_path+'ASVspoof2019_{}_train/'.format(args.track))) Note we bypass the reference to the track to
    # train on LA instead of on DF (there is no provided training or dev data for DF)
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train, labels=d_label_trn,
                                           base_dir=os.path.join(args.database_path + 'ASVspoof2019_LA_train/'),
                                           protocols_dir=os.path.join(
                                               args.protocols_path + '{}_cm_protocols'.format(prefix)),
                                           transform=transforms, is_train=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    del d_label_trn

    # define validation dataloader

    # Note we bypass the reference to the track to validate on LA instead of on DF (there is no provided training or
    # dev data for DF)
    d_label_dev, file_dev = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path + '{}_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'.format(prefix)),
        is_train=False, is_eval=False)
    print('no. of validation trials', len(file_dev))

    # Note we bypass the reference to the track to train on LA instead of on DF (there is no provided training or dev
    # data for DF)
    dev_set = Dataset_ASVspoof2019_train(list_IDs=file_dev,
                                         labels=d_label_dev,
                                         base_dir=os.path.join(args.database_path + 'ASVspoof2019_LA_dev/'),
                                         protocols_dir=os.path.join(
                                             args.protocols_path + '{}_cm_protocols'.format(prefix)),
                                         transform=transforms, is_train=False)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    del dev_set, d_label_dev
    # model
    train_graphs = load_data([train_set.data_x[0]], [train_set.data_y[0]])
    A = nx.to_numpy_matrix(train_graphs[0].g)
    if (args.graph_type == 'cycle'):
        A[0, -1] = 1
        A[-1, 0] = 1
    A = torch.Tensor(A).to(device)
    model = Graph_CNN_ortega(args.num_layers, train_graphs[0].node_features.shape[1],
                             args.hidden_dim, args.num_classes, args.final_dropout, args.graph_pooling_type,
                             device, A)
    model = model.to(device)

    # set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    # evaluation 
    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(
            args.protocols_path + '{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix, prefix_2021)), is_train=False,
            is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(
            args.database_path + 'ASVspoof2021_{}_eval/'.format(args.track)))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    # define train dataloader

    # Training and validation
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 99
    min_EER = 100
    min_EER_epoch = 0

    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader, model, args.lr, optimizer, device)
        valid_accuracy, cm_keys, cm_scores = evaluate_accuracy(dev_loader, model, device)
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_keys == 'spoof']
        eer_cm19 = em.compute_eer(bona_cm, spoof_cm)[0]
        if min_EER > eer_cm19:
            min_EER = eer_cm19
            min_EER_epoch = epoch
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        writer.add_scalar('19-EER', eer_cm19 * 100, epoch)
        print('\nEpoch - {} - loss - {} - train_accuracy - {:.2f} - valid_accuracy - {:.2f}'.format(epoch,
                                                                                                    running_loss,
                                                                                                    train_accuracy,
                                                                                                    valid_accuracy))
        logging.info('Epoch - {} - loss - {} - train_accuracy - {:.2f} - valid_accuracy - {:.2f}'.format(epoch,
                                                                                                         running_loss,
                                                                                                         train_accuracy,
                                                                                                         valid_accuracy))
        print('19 - EER = {:8.5f} %            '
              '            mmin - EER = {:8.5f} % '
              '             epoch = {}'.format(eer_cm19 * 100, min_EER * 100, min_EER_epoch))
        logging.info('19 - EER = {:8.5f} %            '
                     '            mmin - EER = {:8.5f} % '
                     '             epoch = {}'.format(eer_cm19 * 100, min_EER * 100, min_EER_epoch))
        if valid_accuracy > best_acc:
            print('best model find at epoch', epoch)
        best_acc = max(valid_accuracy, best_acc)
        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
