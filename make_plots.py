######################
### Plot functions ###
######################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017

# imports + functions
import matplotlib
import matplotlib.pyplot as plt
import simplejson


def preprocess(list_loss):
    # find overall min value
    min_value = 1000
    for k in range(len(list_loss)):
        min_value = min(list_loss[k]) if (min(list_loss[k]) <= min_value) else min_value

    # subtract min value and add epsilon
    eps = min_value * 1e-6
    for k in range(len(list_loss)):
        list_loss[k] = [i - min_value + eps for i in list_loss[k]]
    return list_loss


def two_d_plot_time(list_loss, list_x, list_params, dataset_name, n, d, log_scale, x_limits=None):
    list_loss = preprocess(list_loss)
    colors = ['#1B2631', '#C0392B', '#9B59B6', '#2980B9', '#1E8449', '#27AE60', '#E67E22', '#95A5A6', '#FF97F2',
              '#34495E']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    fig = plt.figure()

    for i in range(len(list_loss)):
        plt.plot(list_x[i], list_loss[i], linestyles[i % 10], color=colors[i % 10], linewidth=4.0)
    plt.legend(list_params, fontsize=12, loc=1)

    if log_scale == True:
        plt.yscale('log')
        plt.ylabel('$\log(f-f^*)$', fontsize=12)
    else:
        plt.yscale('linear')
        plt.ylabel('$(f-f^*)$')
    plt.xlabel('time in seconds', fontsize=12)
    if not x_limits == None:
        plt.xlim(x_limits)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)
    plt.show


def two_d_plot_iterations(list_loss, list_x, list_params, dataset_name, n, d, log_scale, x_limits=None):
    colors = ['#1B2631', '#C0392B', '#9B59B6', '#2980B9', '#1E8449', '#27AE60', '#E67E22', '#95A5A6', '#FF97F2',
              '#34495E']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']

    fig = plt.figure()

    for i in range(len(list_loss)):
        _x = []
        for k in range(len(list_x[i])):
            _x.append(k)
        list_x[i] = _x

        plt.plot(list_x[i], list_loss[i], linestyles[i % 10], color=colors[i % 10], linewidth=4.0)

    plt.legend(list_params, fontsize=12, loc=1)

    if log_scale == True:
        plt.yscale('log')
        plt.ylabel('$\log(f-f^*)$', fontsize=12)
    else:
        plt.yscale('linear')
        plt.ylabel('$(f-f^*)$')

    plt.xlabel('iteration', fontsize=12)
    if not x_limits == None:
        plt.xlim(x_limits)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)
    plt.show


def two_d_plot_epochs(list_loss, list_samples, list_params, dataset_name, n, d, log_scale, x_limits=None):
    colors = ['#1B2631', '#C0392B', '#9B59B6', '#2980B9', '#1E8449', '#27AE60', '#E67E22', '#95A5A6', '#FF97F2',
              '#34495E']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']

    fig = plt.figure()

    list_x = [[j / n for j in i] for i in list_samples]
    for i in range(len(list_loss)):
        plt.plot(list_x[i], list_loss[i], linestyles[i % 10], color=colors[i % 10], linewidth=4.0)

    plt.legend(list_params, fontsize=12, loc=1)
    if not x_limits == None:
        plt.xlim(x_limits)

    if log_scale == True:
        plt.yscale('log')
        plt.ylabel('$\log(f-f^*)$', fontsize=12)
    else:
        plt.yscale('linear')
        plt.ylabel('$f-f^*$', fontsize=12)

    plt.xlabel('epochs', fontsize=12)
    plt.show
