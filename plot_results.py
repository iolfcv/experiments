import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import CARETDOWN
import argparse
import os

parser = argparse.ArgumentParser(description='Incremental learning')
parser.add_argument('-f', '--file', type=str, 
                    help='CSV file name containing results')

savefig = True
TITLE_SIZE = 28
AXIS_SIZE = 20
matplotlib.rcParams['font.family'] = ['serif']

matplotlib.rc('axes', titlesize=TITLE_SIZE)
matplotlib.rc('axes', labelsize=AXIS_SIZE)
dotted_line_width = 2.5
title = 'CRIB-Toys Incremental Learning Performance'
anchor = (0.65, 0.43)


def plot_acc(datafile):
    fig, ax1 = plt.subplots()
    if not os.path.exists(datafile):
        raise Exception('Results file path not found')

    # Load results from files
    classes_file = '%s-classes.npz' % os.path.splitext(datafile)[0]
    matr_file = '%s-matr.npz' % os.path.splitext(datafile)[0]
    info_classes = np.load(classes_file)
    info_matr = np.load(matr_file)

    classes = info_classes['classes_seen']
    num_classes = info_matr['args'][()].total_classes
    num_le = len(classes) # Number of learning exposures
    
    os.makedirs('results', exist_ok=True)
    filename = 'results/plot.pdf'
    linestyle = '-'
    all_lines = []

    ax1.set_xlabel('Number of learning exposures')
    ax1.set_ylabel('% Test accuracy over seen objects')
    ax1.set_yticks(np.arange(0, 105, 10))
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()
    ax2.set_ylim([0,1.05*num_classes])
    ax2.set_ylabel('Unique objects seen (UOS)')
    ax2.set_yticks(np.arange(0, num_classes+1, num_classes//10))

    plt.xticks(np.arange(0, num_le+1, num_le//10))
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.grid()

    acc_matr = info_matr['acc_matr']
    counter = []
    cnt = 0
    for i in range(len(classes)):
        if classes[i] in classes[:i]:
            counter.append(cnt)
        else:
            cnt += 1
            counter.append(cnt)
    
    test_acc = np.sum(acc_matr, axis=0) / np.array(counter)
    print(test_acc)

    all_lines.append(ax2.plot(np.arange(0, num_le), counter, color='purple', 
                              linestyle=':', linewidth=dotted_line_width, 
                              label='Ground-truth UOS')[0])
    all_lines.append(ax1.plot(np.arange(0, num_le), test_acc, color='blue', 
                              label='Learner Performance', linestyle='-')[0])
                

    all_labels = [l.get_label() for l in all_lines] 
    plt.title(title)
    ax1.legend(all_lines, all_labels, bbox_to_anchor=anchor, 
               loc=2, borderaxespad=0., fancybox=True, framealpha=0.7, 
               fontsize=18, numpoints=1)
        
    plt.gcf().set_size_inches(16, 8)
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                pad_inches=0.01 ,transparent=True)

if __name__ == '__main__':
    args = parser.parse_args()
    plot_acc(args.file)
