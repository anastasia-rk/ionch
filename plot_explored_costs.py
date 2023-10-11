# imports
import numpy as np

from setup import *
import pints
import pickle as pkl
matplotlib.use('AGG')
plt.ioff()

# main
if __name__ == '__main__':
    # read all results from the file
    state_name = 'a'
    with open("Pickles/explore_parameter_space_sequential_" + state_name + ".pkl", "rb") as input_file:
        explore_costs, metadata = pkl.load(input_file)
    # load metadata
    theta_true = metadata['truth']
    keys = ['theta_{' + str(index) + '}' for index in np.arange(len(theta_true))]
    param_names = metadata['param_names']
    nRuns = metadata['nruns']
    # plot cost projections
    nColumns = len(theta_true)
    fig, axes = plt.subplots(2, nColumns, figsize=(12, 8))
    indeces_of_costs_to_plot = [1,4]
    for iKey, key in enumerate(keys):
        range_theta = explore_costs[key][0]
        index_truth = range_theta.index(theta_true[iKey])
        # plot different costs in rows
        for iRow in range(2):
            j = indeces_of_costs_to_plot[iRow]
            for iSample in range(len(range_theta)):
                pl = axes[iRow, iKey].semilogy(explore_costs[key][0][iSample] * np.ones(nRuns), explore_costs[key][j][iSample],
                                       lw=0, color='orange', marker='.',label='cost computed over ' + str(nRuns) + ' runs')
                ind_best_run = explore_costs[key][-1][iSample]
                pl_best = axes[iRow, iKey].semilogy(explore_costs[key][0][iSample], explore_costs[key][j][iSample][ind_best_run],
                                       lw=0, color='purple',marker='.',label='cost at best run')
            ind_best_run = explore_costs[key][-1][index_truth]
            pl_truth = axes[iRow, iKey].semilogy(theta_true[iKey], explore_costs[key][j][index_truth][ind_best_run],
                                   lw=0, color='magenta', marker='o',label='cost at truth')
            axes[iRow, iKey].set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
            if iRow == 0:
                axes[iRow, iKey].set_ylabel(r'$J(C \mid \theta_{' + str(iKey + 1) + r'}, \bar{\mathbf{y}})$')
            else:
                axes[iRow, iKey].set_ylabel(r'$RMSE('+state_name+r'\mid \theta_{' + str(iKey + 1) + r'} \bar{\mathbf{y}})$')
            # assign legend entries only to the last sample and the truth corresponding to the first theta
            if iKey == 0 and iRow == 1:
                lines, legend_labels = axes[iRow,iKey].get_legend_handles_labels()
                axes[iRow,iKey].legend(lines[-3:],legend_labels[-3:],loc='best')
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/costs_projection_inner_RMSE_' + state_name + '.png', dpi=400)

    nColumns = len(theta_true)
    fig, axes = plt.subplots(2, nColumns, figsize=(12, 8))
    indeces_of_costs_to_plot = [2, 3]
    for iKey, key in enumerate(keys):
        range_theta = explore_costs[key][0]
        index_truth = range_theta.index(theta_true[iKey])
        # plot different costs in rows
        for iRow in range(2):
            j = indeces_of_costs_to_plot[iRow]
            for iSample in range(len(range_theta)):
                pl = axes[iRow, iKey].semilogy(explore_costs[key][0][iSample] * np.ones(nRuns),
                                               explore_costs[key][j][iSample],
                                               lw=0, color='orange', marker='.',
                                               label='cost computed over ' + str(nRuns) + ' runs')
                ind_best_run = explore_costs[key][-1][iSample]
                pl_best = axes[iRow, iKey].semilogy(explore_costs[key][0][iSample],
                                                    explore_costs[key][j][iSample][ind_best_run],
                                                    lw=0, color='purple', marker='.', label='cost at best run')
            ind_best_run = explore_costs[key][-1][index_truth]
            pl_truth = axes[iRow, iKey].semilogy(theta_true[iKey],
                                                 explore_costs[key][j][index_truth][ind_best_run],
                                                 lw=0, color='magenta', marker='o', label='cost at truth')
            axes[iRow, iKey].set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
            if iRow == 0:
                axes[iRow, iKey].set_ylabel(r'$G_{data}(\theta_{' + str(iKey + 1) + r'} \mid \bar{\mathbf{y}})$')

            else:
                axes[iRow, iKey].set_ylabel(r'$G_{ODE}(C \mid \theta_{' + str(iKey + 1) + r'}, \bar{\mathbf{y}})$')
            # assign legend entries only to the last sample and the truth corresponding to the first theta
            if iKey == 0 and iRow == 1:
                lines, legend_labels = axes[iRow, iKey].get_legend_handles_labels()
                axes[iRow, iKey].legend(lines[-3:], legend_labels[-3:], loc='best')
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/costs_projection_ODE_data_' + state_name + '.png', dpi=400)

    nColumns = len(theta_true)
    fig, axes = plt.subplots(2, nColumns, figsize=(12, 8))
    indeces_of_costs_to_plot = [5, 6]
    for iKey, key in enumerate(keys):
        range_theta = explore_costs[key][0]
        index_truth = range_theta.index(theta_true[iKey])
        # plot different costs in rows
        for iRow in range(2):
            j = indeces_of_costs_to_plot[iRow]
            for iSample in range(len(range_theta)):
                pl = axes[iRow, iKey].plot(explore_costs[key][0][iSample] * np.ones(nRuns),
                                               explore_costs[key][j][iSample],
                                               lw=0, color='orange', marker='.',
                                               label='cost computed over ' + str(nRuns) + ' runs')
                ind_best_run = explore_costs[key][-1][iSample]
                pl_best = axes[iRow, iKey].plot(explore_costs[key][0][iSample],
                                                    explore_costs[key][j][iSample][ind_best_run],
                                                    lw=0, color='purple', marker='.', label='cost at best run')
            ind_best_run = explore_costs[key][-1][index_truth]
            pl_truth = axes[iRow, iKey].plot(theta_true[iKey],
                                                 explore_costs[key][j][index_truth][ind_best_run],
                                                 lw=0, color='magenta', marker='o', label='cost at truth')
            axes[iRow, iKey].set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
            if iRow == 0:
                axes[iRow, iKey].set_ylabel(r'Evaluations')

            else:
                axes[iRow, iKey].set_ylabel(r'Runtime, s')
            # assign legend entries only to the last sample and the truth corresponding to the first theta
            if iKey == 0 and iRow == 1:
                lines, legend_labels = axes[iRow, iKey].get_legend_handles_labels()
                axes[iRow, iKey].legend(lines[-3:], legend_labels[-3:], loc='best')
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/costs_evals_times_' + state_name + '.png', dpi=400)

    ####################################################################################################################
    ## uncomment this part to plot only one metric
    # nColumns = len(theta_true)
    # fig, axes = plt.subplots(1, nColumns, figsize=(12, 8))
    # j = 5 # index at which the evaluations are stored
    # for iKey, key in enumerate(keys):
    #     range_theta = explore_costs[key][0]
    #     index_truth = range_theta.index(theta_true[iKey])
    #     ax = axes.flatten()[iKey]
    #     for iSample in range(len(range_theta)):
    #         ax.plot(explore_costs[key][0][iSample] * np.ones(nRuns),explore_costs[key][j][iSample],
    #                                        lw=0, color='orange', marker='.',
    #                                        label= str(nRuns) + ' runs')
    #         ind_best_run = explore_costs[key][-1][iSample]
    #         ax.plot(explore_costs[key][0][iSample],explore_costs[key][j][iSample][ind_best_run],
    #                                             lw=0, color='purple', marker='.', label='best run')
    #     ind_best_run = explore_costs[key][-1][index_truth]
    #     ax.plot(theta_true[iKey],explore_costs[key][j][index_truth][ind_best_run],
    #                                          lw=0, color='magenta', marker='o', label='best run at truth')
    #     ax.set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
    #     ax.set_ylabel('number of evaluations per run')
    #     if iKey == 0:
    #         lines, legend_labels = ax.get_legend_handles_labels()
    #         ax.legend(lines[-3:], legend_labels[-3:], loc='best')
    # plt.tight_layout(pad=0.3)
    # plt.savefig('Figures/cost_exploration_evaluations_' + state_name + '.png', dpi=400)
