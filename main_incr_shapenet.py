from model import IncrNet
import torch
from torch.autograd import Variable
from dataset import iDataset
import argparse
import time
import numpy as np
import cv2
import copy
import subprocess
import os
import torch.multiprocessing as mp
import atexit
import sys
import json
from data_generator.shapenet_data_generator import DataGenerator
from collections import deque

parser = argparse.ArgumentParser(description='Incremental learning')

# Saving options
parser.add_argument('--outfile', default='results/temp.csv', type=str,
                    help='Output file name (should have .csv extension)')
parser.add_argument('--save_all', dest='save_all', action='store_true',
                    help='Option to save models after each '
                         'test_freq number of learning exposures')
parser.add_argument('--save_all_dir', dest='save_all_dir', type=str,
                    default=None, help='Directory to store all models in')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='Resume training from checkpoint at outfile')
parser.add_argument('--resume_outfile', default=None, type=str,
                    help='Output file name after resuming')

# Hyperparameters
parser.add_argument('--init_lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--init_lr_ft', default=0.001, type=float,
                    help='Init learning rate for balanced finetuning (for E2E)')
parser.add_argument('--num_epoch', default=15, type=int,
                    help='Number of epochs')
parser.add_argument('--num_epoch_ft', default=10, type=int,
                    help='Number of epochs for balanced finetuning (for E2E)')
parser.add_argument('--lrd', default=10.0, type=float,
                    help='Learning rate decrease factor')
parser.add_argument('--wd', default=0.00001, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--batch_size', default=200, type=int,
                    help='Mini batch size for training')
parser.add_argument('--llr_freq', default=10, type=int,
                    help='Learning rate lowering frequency for SGD (for E2E)')
parser.add_argument('--batch_size_test', default=200, type=int,
                    help='Mini batch size for testing')

# CRIB options
parser.add_argument('--lexp_len', default=100, type=int,
                    help='Number of frames in Learning Exposure')
parser.add_argument('--size_test', default=100, type=int,
                    help='Number of test images per object')
parser.add_argument('--num_exemplars', default=1500, type=int,
                    help='number of exemplars')
parser.add_argument('--img_size', default=224, type=int,
                    help='Size of images input to the network')
parser.add_argument('--rendered_img_size', default=300, type=int,
                    help='Size of rendered images')
parser.add_argument('--total_classes', default=20, type=int,
                    help='Total number of classes')
parser.add_argument('--num_instance_per_class', default=25, type=int, 
                    help='Number of instances per class for training')
parser.add_argument('--num_test_instance_per_class', default=15, type=int, 
                    help='Number of test instances per class')

# Model options
parser.add_argument('--algo', default='icarl', type=str,
                    help='Algorithm to run. Options : icarl, e2e, lwf')
parser.add_argument('--no_dist', dest='dist', action='store_false',
                    help='Option to switch off distillation loss')
parser.add_argument('--pt', dest='pretrained', action='store_true',
                    help='Option to start from an ImageNet pretrained model')
parser.add_argument('--ncm', dest='ncm', action='store_true',
                    help='Use nearest class mean classification (for E2E)')

# Training options
parser.add_argument('--diff_order', dest='d_order', action='store_true',
                    help='Use a random order of classes introduced')
parser.add_argument('--no_jitter', dest='jitter', action='store_false',
                    help='Option for no color jittering (for iCaRL)')
parser.add_argument('--h_ch', default=0.02, type=float,
                    help='Color jittering : max hue change')
parser.add_argument('--s_ch', default=0.05, type=float,
                    help='Color jittering : max saturation change')
parser.add_argument('--l_ch', default=0.1, type=float,
                    help='Color jittering : max lightness change')

# System options
parser.add_argument('--test_freq', default=1, type=int,
                    help='Number of iterations of training after'
                         ' which a test is done/model saved')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Maximum number of threads spawned at any' 
                         'stage of execution')
parser.add_argument('--one_gpu', dest='one_gpu', action='store_true',
                    help='Option to run multiprocessing on 1 GPU')


parser.set_defaults(pre_augment=False)
parser.set_defaults(ncm=False)
parser.set_defaults(dist=True)
parser.set_defaults(pretrained=False)
parser.set_defaults(d_order=False)
parser.set_defaults(jitter=True)
parser.set_defaults(save_all=False)
parser.set_defaults(resume=False)
parser.set_defaults(one_gpu=False)


# Print help if no arguments passed
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
torch.backends.cudnn.benchmark = True
mp.set_sharing_strategy('file_system')
expt_githash = subprocess.check_output(['git', 'describe', '--always'])

# multiprocessing on a single GPU
if args.one_gpu:
    mp.get_context('spawn')

# GPU indices
train_device = 0
test_device = 1
if args.one_gpu:
    test_device = 0

if not os.path.exists(os.path.dirname(args.outfile)):
    os.makedirs(os.path.dirname(args.outfile))

if args.save_all:
    if args.save_all_dir is None:
        raise Exception("Directory to save all model not provided")
    os.makedirs(
        '%s-saved_models' % (os.path.join(
                                args.save_all_dir,
                                (args.outfile).split('.')[0])), 
        exist_ok=True)

# permutation of classes
if args.d_order == True:
    perm_id = np.random.permutation(args.total_classes)
else:
    perm_id = np.arange(args.total_classes)

# NOTE : a permutation file should be present for repeated exposures expt
# this is to ensure that over multiple runs permutations are generated the
# so that all different runs are exposed to the same number of objects
# by any learning exposure
if args.num_instance_per_class > 1:
    perm_file = ('permutation_files/permutation_%d_%d.npy' 
                 % (args.total_classes, args.num_instance_per_class))

    if not os.path.exists(perm_file):
        os.makedirs('permutation_files', exist_ok=True)
        # Create random permutation file and save
        perm_arr = np.array(args.num_instance_per_class 
                            * list(np.arange(args.total_classes)))
        np.random.shuffle(perm_arr)
        np.save(perm_file, perm_arr)
    
    perm_id_all = np.load(perm_file)
    for i in range(len(perm_id_all)):
        perm_id_all[i] = perm_id[perm_id_all[i]]
    perm_id = perm_id_all

# loading mean image; resizing to rendered image size if necessary
mean_image = np.load('data_generator/shapenet_mean_image.npy')
mean_image = cv2.resize(
    mean_image, (args.rendered_img_size, args.rendered_img_size))
mean_image = np.uint8(mean_image)

# To pass to dataloaders for preallocation
max_train_data_size = 2 * args.lexp_len + args.num_exemplars
max_test_data_size = (args.total_classes 
                      * args.num_test_instance_per_class 
                      * args.size_test)

# Initialize CNN
K = args.num_exemplars  # total number of exemplars
model = IncrNet(args, device=train_device)


classes = []
with open('data_generator/shapenet_train_instances.json') as tm_file:
    train_instances = json.load(tm_file)
    for cl in train_instances:
        classes.append(cl)
        tmp_list = []
        for synset, modelID in train_instances[cl]:
            tmp_list.append(modelID)
        train_instances[cl] = tmp_list
classes.sort()
with open('data_generator/shapenet_test_instances.json') as tm_file:
    test_instances = json.load(tm_file)
    for cl in test_instances:
        tmp_list = []
        for synset, modelID in test_instances[cl]:
            tmp_list.append(modelID)
        test_instances[cl] = tmp_list
class_map = {cl_name: idx for idx,
             cl_name in enumerate(classes)}
classes = np.array(classes)
classes_seen = []
instances_seen = []
model_classes_seen = []  # Class index numbers stored by model
exemplar_data = []  # Exemplar set information stored by the model
# acc_matr row index represents class number and column index represents
# learning exposure.
acc_matr = np.zeros((args.total_classes, 
                     args.total_classes * args.num_instance_per_class))


# All object train data generators
data_generators = [deque([]) for i in range(args.total_classes)]
for i in range(args.total_classes):
    instances = np.random.choice(train_instances[classes[i]], 
                                 args.num_instance_per_class, replace=False)
    for instance in instances:
        data_generators[i].append(
            DataGenerator(category_name=classes[i], 
                          instance_name=instance, 
                          n_frames=args.lexp_len, 
                          size_test=args.size_test,
                          resolution=args.rendered_img_size, 
                          job='train'))

test_data_generators = [[] for i in range(args.total_classes)]
for i in range(args.total_classes):
    instances = test_instances[classes[i]]
    for instance in instances:
        test_data_generators[i].append(
            DataGenerator(category_name=classes[i],
                          instance_name=instance,
                          n_frames=args.lexp_len,
                          size_test=args.size_test,
                          resolution=args.rendered_img_size,
                          job='test'))

# Declaring train and test sets
train_set = None
test_set = iDataset(args, mean_image, data_generators=[],
                    max_data_size=max_test_data_size, job='test')

# Conditional variable, shared memory for synchronization
cond_var = mp.Condition()
train_counter = mp.Value('i', 0)
test_counter = mp.Value('i', 0)
dataQueue = mp.Queue()
all_done = mp.Event()
data_mgr = mp.Manager()
expanded_classes = data_mgr.list([None for i in range(args.test_freq)])

if args.resume:
    print('resuming model from %s-model.pth.tar' %
          (args.outfile).split('.')[0])

    model = torch.load('%s-model.pth.tar' % (args.outfile).split('.')
                       [0], map_location=lambda storage, loc: storage)
    model.device = train_device

    model.exemplar_means = []
    model.compute_means = True

    info_classes = np.load('%s-classes.npz' % (args.outfile).split('.')[0])
    info_matr = np.load('%s-matr.npz' % (args.outfile).split('.')[0])
    if expt_githash != info_classes['expt_githash']:
        print('Warning : Code was changed since the last time model was saved')
        print('Last commit hash : ', info_classes['expt_githash'])
        print('Current commit hash : ', expt_githash)

    args_resume_outfile = args.resume_outfile
    perm_id = info_classes['perm_id']
    num_iters_done = info_matr['num_iters_done']
    acc_matr = info_matr['acc_matr']
    args = info_matr['args'].item()

    if args.resume_outfile is not None:
        args.outfile = args.resume_outfile = args_resume_outfile
    else:
        print('Overwriting old files')

    model_classes_seen = list(
        info_classes['model_classes_seen'][:num_iters_done])
    classes_seen = list(info_classes['classes_seen'][:num_iters_done])

    train_counter = mp.Value('i', len(classes_seen))
    test_counter = mp.Value('i', len(classes_seen))

    # expanding test set to everything seen earlier
    for cl in model.classes:
        print('Expanding class for resuming : ', cl)
        test_set.expand(args, [data_generators[class_map[cl]]], 
                        [cl], model.classes_map, 'test')

    # Get the datagenerators state upto resuming point
    for cl in classes_seen:
        data_generators[class_map[cl]].n_exposures += 1

    # Ensuring requires_grad = True after model reload
    for p in model.parameters():
        p.requires_grad = True


def train_run(device):
    global train_set
    if args.algo == 'e2e':
        # Empty train set which would be combined 
        # with exemplars for balanced finetuning
        bf_train_set = iDataset(args, mean_image, data_generators=[], 
                                max_data_size=max_train_data_size,
                                job='train')
    model.cuda(device=device)
    s = len(classes_seen)
    print('####### Train Process Running ########')
    print('Args: ', args)
    train_wait_time = 0

    while s < args.total_classes * args.num_instance_per_class:
        time_ptr = time.time()
        # Do not start training till test process catches up
        cond_var.acquire()
        # while loop to avoid spurious wakeups
        while test_counter.value + args.test_freq <= train_counter.value:
            print('[Train Process] Waiting on test process')
            print('[Train Process] train_counter : ', train_counter.value)
            print('[Train Process] test_counter : ', test_counter.value)
            cond_var.wait()
        cond_var.release()
        train_wait_time += time.time() - time_ptr

        curr_class_idx = perm_id[s]
        curr_class = classes[curr_class_idx]
        classes_seen.append(curr_class)
        # Boolean to store if the current iteration saw a new class
        curr_expanded = False

        # Keep a copy of previous model for distillation
        prev_model = copy.deepcopy(model)
        prev_model.cuda(device=device)
        for p in prev_model.parameters():
            p.requires_grad = False

        if curr_class not in model.classes_map:
            model.increment_classes([curr_class])
            model.cuda(device=device)
            curr_expanded = True

        model_curr_class_idx = model.classes_map[curr_class]
        model_classes_seen.append(model_curr_class_idx)

        # Load Datasets
        print('Loading training examples for'\
              ' class index %d , %s, at iteration %d' % 
              (model_curr_class_idx, curr_class, s))

        if train_set is None:
            train_set = iDataset(args, mean_image, 
                                 [[data_generators[curr_class_idx].popleft()]], 
                                 max_train_data_size, 
                                 [curr_class], model.classes_map, 
                                 'train', le_idx=s)
        else:
            train_set.pseudo_init(args, 
                                  [[data_generators[curr_class_idx].popleft()]], 
                                  [curr_class], model.classes_map, 'train', 
                                  le_idx=s)

        model.train()
        if args.algo == 'icarl' or args.algo == 'lwf':
            model.update_representation_icarl(train_set, 
                                              prev_model, 
                                              [model_curr_class_idx], 
                                              args.num_workers)
        else:
            model.update_representation_e2e(train_set,
                                            prev_model,
                                            args.num_workers,
                                            bft=False)
        model.eval()
        del prev_model

        m = int(K / model.n_classes)

        if args.algo == 'icarl' or args.algo == 'e2e':
            # Reduce exemplar sets for known classes
            model.reduce_exemplar_sets(m)

            # Construct exemplar sets for current class
            print('Constructing exemplar set for class index %d , %s ...' %
                  (model_curr_class_idx, curr_class), end="")

            images, image_means, le_maps, image_bbs = train_set.get_image_class(
                model_curr_class_idx)
            model.construct_exemplar_set(images, image_means, le_maps, 
                                         image_bbs, m, model_curr_class_idx, s)
        

        model.n_known = model.n_classes

        if args.algo == 'e2e':
            bf_train_set.clear()

            prev_model = copy.deepcopy(model)
            prev_model.cuda(device=device)
            for p in prev_model.parameters():
                p.requires_grad = False

            print('E2EIL Balanced Finetuning Phase')
            model.train()
            model.update_representation_e2e(bf_train_set,
                                            prev_model,
                                            args.num_workers,
                                            bft=True)
            model.eval()
            del prev_model

            print('Constructing exemplar set for class index %d , %s ...' %
                  (model_curr_class_idx, curr_class), end="")
            model.construct_exemplar_set(images, image_means, le_maps, 
                                         image_bbs, m, model_curr_class_idx, 
                                         s, overwrite=True)


        print("Model num classes : %d, " % model.n_known)
        
        if args.algo == 'icarl' or args.algo == 'e2e':
            for y, P_y in enumerate(model.exemplar_sets):
                print("Exemplar set for class-%d:" % (y), P_y.shape)

            exemplar_data.append(list(model.eset_le_maps))


        cond_var.acquire()
        train_counter.value += 1
        if curr_expanded:
            expanded_classes[s % args.test_freq] = (curr_class_idx, curr_class)
        else:
            expanded_classes[s % args.test_freq] = None

        if train_counter.value == test_counter.value + args.test_freq:
            temp_model = copy.deepcopy(model)
            temp_model.cpu()
            dataQueue.put(temp_model)
        cond_var.notify_all()
        cond_var.release()

        np.savez('%s-classes.npz' % (args.outfile)[:-4], 
                 model_classes_seen=model_classes_seen,
                 classes_seen=classes_seen, 
                 expt_githash=expt_githash, 
                 exemplar_data=np.array(exemplar_data), perm_id=perm_id)

        # loop var increment
        s += 1

    time_ptr = time.time()
    all_done.wait()
    train_wait_time += time.time() - time_ptr
    print('[Train Process] Done, total time spent waiting : ', train_wait_time)


def test_run(device):
    global test_set
    print('####### Test Process Running ########')
    test_model = None
    s = args.test_freq * (len(classes_seen)//args.test_freq)

    test_wait_time = 0
    with open(args.outfile, 'w') as file:
        print("model classes, Train Accuracy, Test Accuracy", file=file)
        while s < args.total_classes * args.num_instance_per_class:

            # Wait till training is done
            time_ptr = time.time()
            cond_var.acquire()
            while train_counter.value < test_counter.value + args.test_freq:
                print('[Test Process] Waiting on train process')
                print('[Test Process] train_counter : ', train_counter.value)
                print('[Test Process] test_counter : ', test_counter.value)
                cond_var.wait()
            cond_var.release()
            test_wait_time += time.time() - time_ptr

            cond_var.acquire()
            test_model = dataQueue.get()
            expanded_classes_copy = copy.deepcopy(expanded_classes)
            test_counter.value += args.test_freq
            cond_var.notify_all()
            cond_var.release()

            # test set only needs to be expanded
            # when a new exposure is seen
            for expanded_class in expanded_classes_copy:
                if expanded_class is not None:
                    idx, cl = expanded_class
                    print('[Test Process] Loading test data')
                    test_set.expand(args, [test_data_generators[idx]],
                                    [cl], test_model.classes_map, 
                                    'test', args.size_test)

            print("[Test Process] Test Set Length:", test_set.curr_len)
            

            test_model.device = device
            test_model.cuda(device=device)
            test_model.eval()
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=args.batch_size_test, shuffle=False, 
                num_workers=args.num_workers, pin_memory=True)

            print("%d, " % test_model.n_known, end="", file=file)

            print("[Test Process] Computing Accuracy matrix...")

            all_labels = []
            all_preds = []
            with torch.no_grad():
                for indices, images, labels in test_loader:
                    images = Variable(images).cuda(device=device)
                    preds = test_model.classify(images, 
                                                mean_image, 
                                                args.img_size)
                    all_preds.append(preds.data.cpu().numpy())
                    all_labels.append(labels.numpy())
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            for i in range(test_model.n_known):
                class_preds = all_preds[all_labels == i]
                correct = np.sum(class_preds == i)
                total = len(class_preds)
                acc_matr[i, s] = (100.0 * correct/total)

            test_acc = np.mean(acc_matr[:test_model.n_known, s])
            print('%.2f ,' % test_acc, file=file)
            print('[Test Process] =======> Test Accuracy after %d'
                  ' learning exposures : ' %
                  (s + args.test_freq), test_acc)

            print("[Test Process] Saving model and other data")
            test_model.cpu()
            if not args.save_all:
                torch.save(test_model, '%s-model.pth.tar' %
                           (args.outfile).split('.')[0])
            else:
                torch.save(test_model, '%s-saved_models/model_iter_%d.pth.tar' %
                           (os.path.join(args.save_all_dir, 
                                         (args.outfile).split('.')[0]), s))

            # loop var increment
            s += args.test_freq

            np.savez('%s-matr.npz' % (args.outfile).split('.')[0], 
                     acc_matr=acc_matr, 
                     model_hyper_params=model.fetch_hyper_params(), 
                     args=args, num_iters_done=s)

        print("[Test Process] Done, total time spent waiting : ", 
              test_wait_time)
        all_done.set()


def cleanup(train_process, test_process):
    train_process.terminate()
    test_process.terminate()


def main():
    train_process = mp.Process(target=train_run, args=(train_device,))
    test_process = mp.Process(target=test_run, args=(test_device,))
    atexit.register(cleanup, train_process, test_process)
    train_process.start()
    test_process.start()

    train_process.join()
    print('Train Process Completed')
    test_process.join()
    print('Test Process Completed')


if __name__ == "__main__":
    main()
