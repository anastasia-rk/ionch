# imports
import matplotlib.pyplot as plt
import numpy as np

from setup import *
import pints
import pickle as pkl
matplotlib.use('AGG')
plt.ioff()

# main
if __name__ == '__main__':
    # read all results from the file
    lambdas = [1, 10, 100, 1000, 10000, 100000, 1000000]
    plot_colors = ['purple', 'darkorange', 'plum', 'olive', 'magenta', 'peru', 'black']
    plot_markers = ['o', 'X', 'd', '<', '>', 'v', '^']
    state_name = 'r'
    # method_name = 'whole'
    method_name = 'sequential'

    nColumns = 4
    fig1, axes1 = plt.subplots(2, nColumns, figsize=(12, 8))
    fig2, axes2 = plt.subplots(2, nColumns, figsize=(12, 8))
    fig3, axes3 = plt.subplots(2, nColumns, figsize=(12, 8))
    fig4, axes4 = plt.subplots(2, nColumns, figsize=(12, 8))
    fig5, axes5 = plt.subplots(2, nColumns, figsize=(12, 8))
    legend_lines1 = []
    legend_lines2 = []
    legend_lines3 = []
    legend_lines4 = []
    legend_labels1 = []
    legend_labels2 = []
    legend_labels3 = []
    legend_labels4 = []
    for iLambda, lambd in enumerate(lambdas):
        with open("Pickles/explore_parameter_space_" + method_name + "_" + state_name + "_lambda_"+str(int(lambd))+".pkl", "rb") as input_file:
            explore_costs, metadata = pkl.load(input_file)
        # load metadata
        theta_true = metadata['truth']
        keys = ['theta_{' + str(index) + '}' for index in np.arange(len(theta_true))]
        param_names = metadata['param_names']
        nRuns = metadata['nruns']
        # plot cost projections
        # plot stuff in figure one
        indeces_of_costs_to_plot = [1,4]
        for iKey, key in enumerate(keys):
            range_theta = explore_costs[key][0]
            index_truth = range_theta.index(theta_true[iKey])
            # plot different costs in rows
            for iRow in range(2):
                j = indeces_of_costs_to_plot[iRow]
                for iSample in range(len(range_theta)):
                    pl = axes1[iRow, iKey].semilogy(explore_costs[key][0][iSample] * np.ones(nRuns), explore_costs[key][j][iSample],
                                           lw=0, color='orange', marker=plot_markers[iLambda],markersize=3) ## , label='cost computed over ' + str(nRuns) + ' runs'
                    ind_best_run = explore_costs[key][-1][iSample]
                    pl_best = axes1[iRow, iKey].semilogy(explore_costs[key][0][iSample], explore_costs[key][j][iSample][ind_best_run],
                                           lw=0, color=plot_colors[iLambda],marker=plot_markers[iLambda],markersize=3, label='lambda = {:.2e}'.format(lambd))
                ind_best_run = explore_costs[key][-1][index_truth]
                pl_truth = axes1[iRow, iKey].semilogy(theta_true[iKey], explore_costs[key][j][index_truth][ind_best_run],
                                       lw=0, color='dimgray', marker=plot_markers[iLambda],markersize=4,label='value at truth')
                axes1[iRow, iKey].set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
                if iRow == 0:
                    axes1[iRow, iKey].set_ylabel(r'$J(C \mid \theta_{' + str(iKey + 1) + r'}, \bar{\mathbf{y}})$')
                else:
                    axes1[iRow, iKey].set_ylabel(r'$RMSE('+state_name+r'\mid \theta_{' + str(iKey + 1) + r'}, \bar{\mathbf{y}})$')
                # assign legend entries only to the last sample and the truth corresponding to the first theta
                if iKey == 0 and iRow == 1:
                    lines, legend_labels = axes1[iRow,iKey].get_legend_handles_labels()
                    legend_lines1 = legend_lines1 + lines[-2:]
                    legend_labels1 = legend_labels1 + legend_labels[-2:]
                    # axes1[iRow,iKey].legend(lines[-2:],legend_labels[-2:],loc='best')

        # plot stuff in figure two
        indeces_of_costs_to_plot = [2, 3]
        for iKey, key in enumerate(keys):
            range_theta = explore_costs[key][0]
            index_truth = range_theta.index(theta_true[iKey])
            # plot different costs in rows
            for iRow in range(2):
                j = indeces_of_costs_to_plot[iRow]
                for iSample in range(len(range_theta)):
                    pl = axes2[iRow, iKey].semilogy(explore_costs[key][0][iSample] * np.ones(nRuns),
                                                   explore_costs[key][j][iSample],
                                                   lw=0, color='orange', marker=plot_markers[iLambda],markersize=3)
                    ind_best_run = explore_costs[key][-1][iSample]
                    pl_best = axes2[iRow, iKey].semilogy(explore_costs[key][0][iSample],
                                                        explore_costs[key][j][iSample][ind_best_run],
                                                        lw=0, color=plot_colors[iLambda], marker=plot_markers[iLambda],markersize=3, label='lambda = {:.2e}'.format(lambd))
                ind_best_run = explore_costs[key][-1][index_truth]
                pl_truth = axes2[iRow, iKey].semilogy(theta_true[iKey],
                                                     explore_costs[key][j][index_truth][ind_best_run],
                                                     lw=0, color='dimgray', marker=plot_markers[iLambda],markersize=4, label='value at truth')
                axes2[iRow, iKey].set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
                if iRow == 0:
                    axes2[iRow, iKey].set_ylabel(r'$G_{data}(\theta_{' + str(iKey + 1) + r'} \mid \bar{\mathbf{y}})$')

                else:
                    axes2[iRow, iKey].set_ylabel(r'$G_{ODE}(C \mid \theta_{' + str(iKey + 1) + r'}, \bar{\mathbf{y}})$')
                # assign legend entries only to the last sample and the truth corresponding to the first theta
                if iKey == 0 and iRow == 1:
                    lines, legend_labels = axes2[iRow, iKey].get_legend_handles_labels()
                    legend_lines2 = legend_lines2 + lines[-2:]
                    legend_labels2 = legend_labels2 + legend_labels[-2:]
                    # axes2[iRow, iKey].legend(lines[-2:], legend_labels[-2:], loc='best')
        # plot stuff in figure three
        indeces_of_costs_to_plot = [5, 6]
        for iKey, key in enumerate(keys):
            range_theta = explore_costs[key][0]
            index_truth = range_theta.index(theta_true[iKey])
            # plot different costs in rows
            for iRow in range(2):
                j = indeces_of_costs_to_plot[iRow]
                for iSample in range(len(range_theta)):
                    pl = axes3[iRow, iKey].plot(explore_costs[key][0][iSample] * np.ones(nRuns),
                                                   explore_costs[key][j][iSample],
                                                   lw=0, color='orange', marker=plot_markers[iLambda],markersize=3,
                                                   label='cost computed over ' + str(nRuns) + ' runs')
                    ind_best_run = explore_costs[key][-1][iSample]
                    pl_best = axes3[iRow, iKey].plot(explore_costs[key][0][iSample],
                                                        explore_costs[key][j][iSample][ind_best_run],
                                                        lw=0, color=plot_colors[iLambda], marker=plot_markers[iLambda],markersize=3,  label='lambda = {:.2e}'.format(lambd))
                ind_best_run = explore_costs[key][-1][index_truth]
                pl_truth = axes3[iRow, iKey].plot(theta_true[iKey],
                                                     explore_costs[key][j][index_truth][ind_best_run],
                                                     lw=0, color='dimgray', marker=plot_markers[iLambda],markersize=4, label='value at truth')
                axes3[iRow, iKey].set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
                if iRow == 0:
                    axes3[iRow, iKey].set_ylabel(r'Evaluations')

                else:
                    axes3[iRow, iKey].set_ylabel(r'Runtime, s')
                # assign legend entries only to the last sample and the truth corresponding to the first theta
                if iKey == 0 and iRow == 1:
                    lines, legend_labels = axes3[iRow, iKey].get_legend_handles_labels()
                    legend_lines3 = legend_lines3 + lines[-2:]
                    legend_labels3 = legend_labels3 + legend_labels[-2:]
                    # axes3[iRow, iKey].legend(lines[-2:], legend_labels[-2:], loc='best')
        #################################################################################################################
        # plot slopes of costs in figure four
        indeces_of_costs_to_plot = [2,1]
        for iKey, key in enumerate(keys):
            range_theta = explore_costs[key][0]
            dTheta = range_theta[1] - range_theta[0] # compute the step size - keep in mind we had regular intervals
            index_truth = range_theta.index(theta_true[iKey])
            # plot different costs in rows
            for iRow in range(2):
                j = indeces_of_costs_to_plot[iRow]
                for iSample in range(len(range_theta)):
                    # compute the slope of the cost
                    if iSample < index_truth:
                        slope = explore_costs[key][j][iSample+1] - explore_costs[key][j][iSample] / dTheta
                        pl = axes4[iRow, iKey].semilogy(explore_costs[key][0][iSample] * np.ones(nRuns), abs(slope),
                                           lw=0, color='orange', marker=plot_markers[iLambda],markersize=3)
                        ind_best_run = explore_costs[key][-1][iSample]
                        pl_best = axes4[iRow, iKey].semilogy(explore_costs[key][0][iSample], abs(slope[ind_best_run]),
                                           lw=0, color=plot_colors[iLambda],marker=plot_markers[iLambda],markersize=3, label='lambda = {:.2e}'.format(lambd))
                    if iSample > index_truth:
                        slope = explore_costs[key][j][iSample] - explore_costs[key][j][iSample-1] / dTheta
                        pl = axes4[iRow, iKey].semilogy(explore_costs[key][0][iSample] * np.ones(nRuns), abs(slope),
                                           lw=0, color='orange', marker=plot_markers[iLambda],markersize=3)
                        ind_best_run = explore_costs[key][-1][iSample]
                        pl_best = axes4[iRow, iKey].semilogy(explore_costs[key][0][iSample], abs(slope[ind_best_run]),
                                           lw=0, color=plot_colors[iLambda],marker=plot_markers[iLambda],markersize=3, label='lambda = {:.2e}'.format(lambd))
                # get the height of the y axes
                yLim = axes4[iRow, iKey].get_ylim()
                # plot the vertical line at the true theta
                if iLambda == len(lambdas)-1:
                    pl_truth = axes4[iRow, iKey].plot(theta_true[iKey] * np.ones(2), yLim, lw=1, color='dimgray', label='truth')
                # pl_truth = axes4[iRow, iKey].semilogy(theta_true[iKey], 1e-10,
                #                        lw=0, color='dimgray', marker=plot_markers[iLambda],markersize=4,label='value at truth')
                axes4[iRow, iKey].set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
                if iRow == 0:
                    axes4[iRow, iKey].set_ylabel(r'$| \partial  G_{data}(\theta_{' + str(iKey + 1) + r'} \mid \bar{\mathbf{y}})/ \partial\theta_{' + str(iKey + 1) + r'}|$')
                else:
                    axes4[iRow, iKey].set_ylabel(r'$| \partial J(C \mid \theta_{' + str(iKey + 1) + r'}, \bar{\mathbf{y}})/ \partial\theta_{' + str(iKey + 1) + r'}|$')
                # assign legend entries only to the last sample and the truth corresponding to the first theta
                if iKey == 0 and iRow == 1:
                    lines, legend_labels = axes4[iRow,iKey].get_legend_handles_labels()
                    if iLambda == len(lambdas) - 1:
                        legend_lines4 = legend_lines4 + lines[-2:]
                        legend_labels4 = legend_labels4 + legend_labels[-2:]
                    else:
                        legend_lines4 = legend_lines4 + lines[-1:]
                        legend_labels4 = legend_labels4 + legend_labels[-1:]
    # #         plot stuff in figure five
    #     indeces_of_costs_to_plot = [2, 3]
    #     for iKey, key in enumerate(keys):
    #         range_theta = explore_costs[key][0]
    #         dTheta = range_theta[1] - range_theta[0]  # compute the step size - keep in mind we had regular intervals
    #         index_truth = range_theta.index(theta_true[iKey])
    #         # plot different costs in rows
    #         for iRow in range(2):
    #             j = indeces_of_costs_to_plot[iRow]
    #             for iSample in range(len(range_theta)):
    #                 # compute the slope of the cost
    #                 if iSample < index_truth:
    #                     slope = explore_costs[key][j][iSample + 1] - explore_costs[key][j][iSample] / dTheta
    #                     pl = axes5[iRow, iKey].plot(explore_costs[key][0][iSample] * np.ones(nRuns), slope,
    #                                                     lw=0, color='orange', marker=plot_markers[iLambda], markersize=3)
    #                     ind_best_run = explore_costs[key][-1][iSample]
    #                     pl_best = axes5[iRow, iKey].plot(explore_costs[key][0][iSample], slope[ind_best_run],
    #                                                          lw=0, color=plot_colors[iLambda], marker=plot_markers[iLambda],
    #                                                          markersize=3, label='lambda = {:.2e}'.format(lambd))
    #                 if iSample > index_truth:
    #                     slope = explore_costs[key][j][iSample] - explore_costs[key][j][iSample - 1] / dTheta
    #                     pl = axes5[iRow, iKey].plot(explore_costs[key][0][iSample] * np.ones(nRuns), slope,
    #                                                     lw=0, color='orange', marker=plot_markers[iLambda], markersize=3)
    #                     ind_best_run = explore_costs[key][-1][iSample]
    #                     pl_best = axes5[iRow, iKey].plot(explore_costs[key][0][iSample], slope[ind_best_run],
    #                                                          lw=0, color=plot_colors[iLambda], marker=plot_markers[iLambda],
    #                                                          markersize=3, label='lambda = {:.2e}'.format(lambd))
    #             # get the height of the y axes
    #             yLim = axes5[iRow, iKey].get_ylim()
    #             # plot the vertical line at the true theta
    #             if iLambda == len(lambdas) - 1:
    #                 pl_truth = axes5[iRow, iKey].plot(theta_true[iKey] * np.ones(2), yLim, lw=1, color='dimgray',
    #                                                   label='truth')
    ## end loop over lambdas
    ####################################################################################################################
    # set up legend for each figure
    nColumns = int(np.ceil((len(lambdas)+1)/2))
    iKey, iRow = 0, 0
    axes3[iRow, iKey].legend(legend_lines3, legend_labels3,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                borderaxespad=0,ncol=nColumns)
    axes2[iRow, iKey].legend(legend_lines2, legend_labels2,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                borderaxespad=0,ncol=nColumns)
    axes1[iRow, iKey].legend(legend_lines1, legend_labels1,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                borderaxespad=0,ncol=nColumns)
    # legend entries for figs 4 and 5 are shared
    axes4[iRow, iKey].legend(legend_lines4, legend_labels4,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                borderaxespad=0,ncol=nColumns)
    # axes5[iRow, iKey].legend(legend_lines4, legend_labels4, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #                          borderaxespad=0, ncol=nColumns)
    ## save figures for all lambdas
    plt.figure(fig1.number)
    # plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.savefig('Figures/costs_projection_inner_RMSE_' + method_name + '_' + state_name + '.png', dpi=400)
    plt.figure(fig2.number)
    # plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.savefig('Figures/costs_projection_ODE_data_' + method_name + '_' + state_name + '.png', dpi=400)
    # switch current figure to fig1 and save it to png file
    plt.figure(fig3.number)
    # plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.savefig('Figures/costs_evals_times_'  + method_name + '_' + state_name + '.png', dpi=400)
    plt.figure(fig4.number)
    # plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.savefig('Figures/cost_slopes_log_' + method_name + '_' + state_name + '.png', dpi=400)
    # plt.figure(fig5.number)
    # # plt.tight_layout(pad=0.3)
    # plt.subplots_adjust(wspace=0.4, hspace=0.2)
    # plt.savefig('Figures/cost_slopes_decimal_' + method_name + '_' + state_name + '.png', dpi=400)
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
    #                                          lw=0, color='dimgray', marker='o', label='best run at truth')
    #     ax.set_xlabel(r'$\theta_{' + str(iKey + 1) + '} = log(' + param_names[iKey] + ')$')
    #     ax.set_ylabel('number of evaluations per run')
    #     if iKey == 0:
    #         lines, legend_labels = ax.get_legend_handles_labels()
    #         ax.legend(lines[-2:], legend_labels[-2:], loc='best')
    # plt.tight_layout(pad=0.3)
    # plt.savefig('Figures/cost_exploration_evaluations_' + method_name + '_' + state_name + '.png', dpi=400)
