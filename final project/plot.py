import matplotlib.pyplot as plt
import numpy as np
global fig_index
fig_index = 1
def plot(x, y, title, y_label, legend, plot_style = '-', fig = 1):
    global fig_index
    plt.plot(x, y, plot_style)
    plt.title(title)
    plt.legend(legend)
    plt.xlabel('Epoches')
    plt.ylabel(y_label)
    #plt.show()
    
    plt.savefig('figures/res_' + str(fig_index) + '.jpg')
    plt.cla()
    fig_index += 1

def plot_group(groups, title, y_label, legend, plot_style = '-', fig = 1, ps = None):
    global fig_index
    for i, xy in enumerate(groups):
        if ps is None:
            plt.plot(xy[0], xy[1], plot_style)
        else:
            plt.plot(xy[0], xy[1], ps[i])
    plt.title(title)
    plt.legend(legend)
    plt.xlabel('Epoches')
    if y_label == 'max tile':
        plt.yticks([128, 256, 512, 1024], (128, 256, 512, 1024))
    plt.ylabel(y_label)
    #plt.show()

    plt.savefig('figures/res_' + str(fig_index) + '.jpg')
    plt.cla()
    fig_index += 1
def read_and_plot(filename, plot_list, title = 'None', limited = float('inf'), isPlot = False):
    f = open(filename, 'r')
    lines = f.readlines()
    acc = []
    acc_i = 1
    acc_x = []

    test = []
    test_i = 1
    test_x = []   

    max_tile = []
    max_tile_i = 1
    max_tile_x = [] 

    loss_vae = []
    loss_vae_i = 1
    loss_vae_x = [] 

    loss_dqn = []
    loss_dqn_i = 1
    loss_dqn_x = [] 

    loss = []
    loss_i = 1
    loss_x = []
    for i, l in enumerate(lines):
        
        if 'acc' in l:

            index_acc = l.index('acc:')
            acc_item = float(l[index_acc + 5: index_acc + 10])
            acc.append(acc_item)
            acc_x.append(50 * acc_i)
            acc_i += 1

        if 'test score: ' in l:

            test_index = l.index('test score:')
            max_index = l.index(', max')
            #print(l[test_index + 12: max_index])
            test_item = float(l[test_index + 12: max_index])

            test.append(test_item)
            test_x.append(50 * test_i)
            test_i += 1

        if 'max' in l:

            max_tile_index = l.index('max')
            #print(l[test_index + 12: max_index])
            max_tile_item = float(l[max_tile_index + 10: max_tile_index + 10 + 5])

            max_tile.append(max_tile_item)
            max_tile_x.append(50 * max_tile_i)
            max_tile_i += 1


        if 'loss_vae' in l:

            loss_vae_index = l.index('loss_vae')
            #print(l[test_index + 12: max_index])
            loss_vae_item = float(l[loss_vae_index + 10: loss_vae_index + 10 + 8])

            loss_vae.append(loss_vae_item)
            loss_vae_x.append(loss_vae_i)
            loss_vae_i += 1


        if 'loss_dqn' in l:

            loss_dqn_index = l.index('loss_dqn')
            #print(l[test_index + 12: max_index])
            loss_dqn_item = float(l[loss_dqn_index + 10: loss_dqn_index + 10 + 8])

            loss_dqn.append(loss_dqn_item)
            loss_dqn_x.append(loss_dqn_i)
            loss_dqn_i += 1

        if 'loss:' in l:

            loss_index = l.index('loss')
            #print(l[test_index + 12: max_index])
            loss_item = float(l[loss_index + 6: loss_index + 6 + 8])

            loss.append(loss_item)
            loss_x.append(loss_i)
            loss_i += 1
        if i > limited:
            break
    f.close()
    if '1' in plot_list:
        acc_x = (np.array(acc_x) / 50).tolist()
        test_x = (np.array(test_x) / 50).tolist()
        max_tile_x = (np.array(max_tile_x) / 50).tolist()
    if isPlot:
        if 'acc' in plot_list:
            plot(acc_x, acc, title, 'accuracy', ['accuracy'], fig = 1)
        if 'acc_1' in plot_list:
            acc_x = (np.array(acc_x) / 50).tolist()
            plot(acc_x, acc, title, 'accuracy', ['accuracy'], fig = 2)
        if 'test score' in plot_list:
            plot(test_x, test, title, 'test score', ['test score'], fig = 3)
        if 'max tile' in plot_list:
            plot(max_tile_x, max_tile, title, 'max tile', ['max tile'], plot_style = 'o', fig = 4)
        if 'loss vae' in plot_list:
            plot(loss_vae_x, loss_vae, title, 'loss vae', ['loss vae'], fig = 5)
        if 'loss dqn' in plot_list:
            plot(loss_dqn_x, loss_dqn, title, 'loss dqn', ['loss dqn'], fig = 6)
        if 'loss' in plot_list:
            plot(loss_x, loss, title, 'loss', ['loss'], fig = 7)
        if 'ls' in plot_list:
            ls = (np.array(loss_dqn) + np.array(loss_vae)).tolist()
            plot(loss_dqn_x, ls, title, 'total loss', ['total loss'], fig = 8)
    return  (acc_x, acc), (test_x, test), (max_tile_x, max_tile), (loss_vae_x, loss_vae), (loss_dqn_x, loss_dqn), (loss_x, loss)
#read_and_plot('result_archive/result_vae_dqn.txt', ['acc', 'test score', 'loss vae', 'loss dqn', 'max tile'])
#read_and_plot('results/result_dqn_vanila.txt', ['test score','max tile', 'loss'])
#read_and_plot('results/result_vae.txt', ['acc_1', 'loss'], 'Vanilla VAE with Random agent', limited = 10000)
#read_and_plot('results/result_vae.txt', ['acc', 'loss'], limited = 10000)
acc_vae_dqn, test_score_vae_dqn, max_tile_vae_dqn, loss_vae_vae_dqn, loss_dqn_vae_dqn, loss_vae_dqn = read_and_plot(
'result_archive/result_vae_dqn.txt', ['acc', 'test score', 'loss vae', 'loss dqn', 'max tile', 'ls'], title = 'Variational Learning DQN', isPlot = True)

_, test_score_DQN, max_tile_DQN, _, _, loss_DQN = read_and_plot(
'result_archive/result_dqn_vanila.txt', ['test score', 'max tile', 'loss'], title = 'dqn_v')

_, test_score_random, max_tile_random, _, _, _ = read_and_plot(
'result_archive/random.txt', ['test score','max tile'], title = 'random', limited = 600)

acc_vae, _, _, _, _, loss_vae = read_and_plot(
'result_archive/result_vae.txt', ['acc', 'loss', '1'], title = 'vae_v', limited = 14500)

#plot_group(groups, title, y_label, legend, plot_style = '-', fig = 1)
plot_group([acc_vae, acc_vae_dqn], 'Accuracy of Vanilla VAE and VL DQN', 'accuracy', ['Vanilla VAE','VL DQN'], plot_style = '-', fig = 9)
plot_group([test_score_random, test_score_DQN, test_score_vae_dqn], 'Test Score among three approches', 'test score', ['Random','Vanilla DQN', 'VL DQN'], plot_style = '-', fig = 10)
plot_group([max_tile_random, max_tile_DQN, max_tile_vae_dqn], 'Max Tile among three approches', 'max tile', ['Random','Vanilla DQN', 'VL DQN'], plot_style = '*', fig = 11, ps = ['1', '2', '3'])
plot_group([loss_vae, loss_vae_vae_dqn], 'Loss of Vanilla VAE and VL DQN ', 'loss', ['Vanilla VAE','VL DQN'], plot_style = '-', fig = 12)
plot_group([loss_DQN, loss_dqn_vae_dqn], 'Loss of Vanilla DQN and VL DQN ', 'loss', ['Vanilla DQN','VL DQN'], plot_style = '-', fig = 13)