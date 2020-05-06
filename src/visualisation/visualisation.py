import matplotlib

matplotlib.use('Agg')
import mne
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np



def plot_R_multi_measure(frames, x, y, nrow, ncol, title, sub_titles):
    """ plots a boxplot of a subplot determined by the nrow and ncol parameters

    Parameters
    ----------
    folder_path: string
        A path to the folder conrtaing the dataframes that conatin the data
        to be plotted

    x: str
        string to denote the column in the dataframe for the x axis

    y: str
        string top denote the column in the dataframe for the y axis

    ncol: int
        integer staing the number of column in the subplots

    nrow: int
        integer stating the number of rows in the subplots

    title: string
        super title for the figure

    sub_titles: list
        list of strings for the title of the subplots

    """
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    column = 0
    row = 0

    for data_frame in frames:
        if (column > (ncol - 1)):
            column = 0
            row += 1
        ax = axs[row][column]

        data_frame.rename(columns={x:'Language Model', y:"Adjusted R Squared"}, inplace=True)

        print(data_frame)
        sns.boxplot(x='Language Model', y="Adjusted R Squared", data=data_frame, palette="hls", ax=ax)

        if(column ==1 and row == 0):
            ax.set_ylim(top=1)
            ax.set_ylabel('Accuracy')

        else:
            ax.set_ylim(top=0.15)
        ax.set_xticklabels(['Base', 'GPT-2', 'GPT-2-XL', 'CTRL', 'TXL'])
        ax.title.set_text(sub_titles.pop())
        column += 1

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout(pad=1.2, w_pad=0.1, h_pad=0)
    plt.subplots_adjust(left=0.05, right=0.95)
    fig.suptitle(title, weight='bold')
    return fig


def r_squared_boxplot(data_frame, x, y, title, output_path):
    """ plots a boxplot of the data in the data_frame parameter

        Parameters
        ----------
        data_frame: pandas Dataframe
            Data frame consisting of two columns: Model type, Metric

        x: str
            string to denote the column in the dataframe for the x axis

        y: str
            string top denote the column in the dataframe for the y axis

    """
    fig = plt.figure(figsize=(8, 8))
    sns.boxplot(x=x, y=y, data=data_frame, palette="hls")
    fig.suptitle(title)
    plt.savefig(output_path + title + '.png')
    return fig


def plot_rERP(Evoked_dict, output_file):
    """ plots a boxplot of the data in the data_frame parameter

    Parameters
    ----------
    Evoked_dict: dictionary
       Dictionary consisting of mne Evoked Arrays

    output_file: str
       filepath string to save the plot to

    """
    for Evoked_object in Evoked_dict:
        evoked_array= Evoked_dict[Evoked_object]
        evoked_array.nave = 0

        # PLotting
        Title = Evoked_object+ ' Coefficients over time window'
        ts_args = {'scalings':{'eeg':1},'spatial_colors':True, 'units':'β - coefficients', 'titles':Title}
        topomap_args = {'scalings':{'eeg':1}}
        evoked_array.plot_joint(ts_args=ts_args, topomap_args= topomap_args)

        # save figure
        plt.savefig(output_file+ Evoked_object+ '.png')

    return plt

def animated_topographical_maps(coefficinets, start, stop, sampling_frequency):
    step = (stop - start)/sampling_frequency
    time_lags = np.arange(start, stop, step=step)
    Surprisal = mne.Evoked('Results/EEG_Data/Temporal_Response/Individual/GPT2XL/Surprisal_ERP-ave.fif')
    Surprisal.animate_topomap(ch_type='eeg', time_unit='ms', frame_rate=60, butterfly=True, times=time_lags)
    anim.save('topo.gif', writer='imagemagick')



    """
    folder_path = '/Users/markfriel/Documents/Year_4/Project/research_project/Results/Reading_Data/Individual_subjects/R_Score_Eye_measure/'
    frames = util.load_dataframes(folder_path)
    sub_titles = ['First Fixation Duration', 'Gaze Duration', 'Word Skip', 'Total Reading Time']
    title = 'Explanatory value of  language models surprisal for eye-tracking inspections'
    _ = plot_R_multi_measure(frames, "metric", "Adjusted_R_Squared", 2, 2, title, sub_titles)


    # Coefficients over time plot
    Surprisal = mne.Evoked('Results/EEG_Data/Temporal_Response/Individual/GPT2XL/Surprisal_ERP-ave.fif')
    Word_onset = mne.Evoked('Results/EEG_Data/Temporal_Response/Individual/GPT2XL/Word_onset_ERP-ave.fif')

    fig, axs = plt.subplots(nrows=2)
    ax1 = axs[0]
    Surprisal.apply_baseline().plot(scalings={'eeg': 1}, spatial_colors=True, units='β - coefficients', axes=[axs[0]],
                                    titles='Surprisal Coeficients')
    Word_onset.apply_baseline().plot(scalings={'eeg': 1}, spatial_colors=True, units='β - coefficients', axes=[axs[1]],
                                     titles='Word onset Coeficients')
    fig.tight_layout(rect=[-.05, 0.03, 1, 0.95])
    fig.suptitle('GPT2-XL Coefficients from Time Resolved Regression')
    fig.savefig('Topographic_plot.png', dpi=600)


    # Topography Plots
    CTRL_Surprisal = mne.Evoked('Results/EEG_Data/Temporal_Response/Individual/CTRL/Surprisal_ERP-ave.fif')
    GPT2_Surprisal = mne.Evoked('Results/EEG_Data/Temporal_Response/Individual/GPT2/Surprisal_ERP-ave.fif')
    TXL_Surprisal = mne.Evoked('Results/EEG_Data/Temporal_Response/Individual/TXL/Surprisal_ERP-ave.fif')

    fig, axs = plt.subplots(nrows=4, ncols=4)
    GPT2_Surprisal.plot_topomap(scalings={'eeg': 1}, times=[.06, .4, .65, .8],
                                axes=[axs[0][0], axs[0][1], axs[0][2], axs[0][3]], colorbar=False)
    axs[0][0].set_ylabel("GPT2")
    Surprisal.plot_topomap(scalings={'eeg': 1}, times=[.06, .4, .65, .8],
                           axes=[axs[1][0], axs[1][1], axs[1][2], axs[1][3]], colorbar=False)
    axs[1][0].set_ylabel("GPT2-XL")
    TXL_Surprisal.plot_topomap(scalings={'eeg': 1}, times=[.06, .4, .65, .8],
                               axes=[axs[2][0], axs[2][1], axs[2][2], axs[2][3]], colorbar=False)
    axs[2][0].set_ylabel("Transformer-XL")
    CTRL_Surprisal.plot_topomap(scalings={'eeg': 1}, times=[.06, .4, .65, .8],
                                axes=[axs[3][0], axs[3][1], axs[3][2], axs[3][3]], colorbar=False)
    axs[3][0].set_ylabel("CTRL")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.suptitle('Topographies of Langugae Model Coefficients')
    fig.savefig('Topographic_plot.png', dpi=600)
    """

if __name__ =='__main__':
    time_lags = np.arange(-0.2 , .7, step=0.009375)
    Surprisal = mne.Evoked('Results/EEG_Data/Temporal_Response/Individual/GPT2XL/Surprisal_ERP-ave.fif')
    fig, anim = Surprisal.animate_topomap(ch_type='eeg', time_unit='ms', frame_rate=3, times=time_lags, show=False)
