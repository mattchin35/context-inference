#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob
import re
from datetime import datetime
import matplotlib.ticker as mticker
from itertools import cycle
from tkinter import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
from collections import deque
from scipy.stats import sem
from os import listdir
import math

def generate_daily_raster(name_of_cleaned_csv_file): #start_time_sec and end_time_sec are for subsetting the raster plot
    # get_ipython().run_line_magic('matplotlib', 'inline')
    cleaned_csv = pd.read_csv(name_of_cleaned_csv_file)
    replacement_list = ['inactive_press', 'active_press', 'reward_left', 'reward_right', 'lick_left', 'lick_right', 'session_begin', 'ContextA', 'ContextB', 'ContextC']    
    
    inactive_press_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'inactive_press')]
    active_press_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'active_press')]
    reward_left_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'reward_left')]
    reward_right_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'reward_right')]
    lick_left_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'lick_left')]
    lick_right_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'lick_right')]
    ContextA_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'ContextA')]
    ContextB_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'ContextB')]
    ContextC_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'ContextC')]

    #this creates np arrays for each output
    inactive_press_timestamp_array = np.array(inactive_press_df['Time']) 
    active_press_timestamp_array = np.array(active_press_df['Time']) 
    reward_left_timestamp_array = np.array(reward_left_df['Time']) 
    reward_right_timestamp_array = np.array(reward_right_df['Time']) 
    lick_left_timestamp_array = np.array(lick_left_df['Time'])
    lick_right_timestamp_array = np.array(lick_right_df['Time']) 
    ContextA_timestamp_array = np.array(ContextA_df['Time'])
    ContextB_timestamp_array = np.array(ContextB_df['Time'])
    ContextC_timestamp_array = np.array(ContextC_df['Time'])                                     

    #start_time_sec and end_time_sec get added to this list via the GUI; list cleared each time a new plot is created so it must be repopulated with a start and end time in order to properly create a new plot
    time_list =[]
    
    #this plots the raster plot with abridged time between start_time and end_time; make end_time giant if you want unabridged data                                                    
    def new_plot(): #creates a new raster plot for each press of the create_new_plot option in the GUI
        label.config(text = 'raster plot created') #shows raster plot created in the GUI window each time a plot is created
        inactive_press_timestamp_array_abridged = inactive_press_timestamp_array[(inactive_press_timestamp_array >= time_list[0]*1000) & (inactive_press_timestamp_array <= time_list[1]*1000)]
        active_press_timestamp_array_abridged = active_press_timestamp_array[(active_press_timestamp_array >= time_list[0]*1000) & (active_press_timestamp_array <= time_list[1]*1000)]
        reward_left_timestamp_array_abridged = reward_left_timestamp_array[(reward_left_timestamp_array >= time_list[0]*1000) & (reward_left_timestamp_array <= time_list[1]*1000)]
        reward_right_timestamp_array_abridged = reward_right_timestamp_array[(reward_right_timestamp_array >= time_list[0]*1000) & (reward_right_timestamp_array <= time_list[1]*1000)]
        lick_left_timestamp_array_abridged = lick_left_timestamp_array[(lick_left_timestamp_array >= time_list[0]*1000) & (lick_left_timestamp_array <= time_list[1]*1000)]
        lick_right_timestamp_array_abridged = lick_right_timestamp_array[(lick_right_timestamp_array >= time_list[0]*1000) & (lick_right_timestamp_array <= time_list[1]*1000)]
        ContextA_timestamp_array_abridged = ContextA_timestamp_array[(ContextA_timestamp_array >= time_list[0]*1000) & (ContextA_timestamp_array <= time_list[1]*1000)] #updated to include contexts in raster plot
        ContextB_timestamp_array_abridged = ContextB_timestamp_array[(ContextB_timestamp_array >= time_list[0]*1000) & (ContextB_timestamp_array <= time_list[1]*1000)]
        ContextC_timestamp_array_abridged = ContextC_timestamp_array[(ContextC_timestamp_array >= time_list[0]*1000) & (ContextC_timestamp_array <= time_list[1]*1000)]
        
        time_list.clear() #clears the time_list for the next plot creation

        #this creates an np array of np arrays, which is suitable for eventplot
        # array_of_timestamp_arrays = np.array([inactive_press_timestamp_array_abridged, active_press_timestamp_array_abridged,
        #                                       reward_left_timestamp_array_abridged, reward_right_timestamp_array_abridged,
        #                                       lick_left_timestamp_array_abridged, lick_right_timestamp_array_abridged,
        #                                       ContextA_timestamp_array_abridged, ContextB_timestamp_array_abridged, ContextC_timestamp_array_abridged])

        array_of_timestamp_arrays = np.array(
            [active_press_timestamp_array_abridged,
             reward_left_timestamp_array_abridged, reward_right_timestamp_array_abridged,
             lick_left_timestamp_array_abridged, lick_right_timestamp_array_abridged])

        #this creates the raster plot
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(8)
        fig.set_figwidth(16)
        ax.eventplot(array_of_timestamp_arrays, linelengths = 0.5, linewidths=0.5,color='black')
        ax.set_title('Behavioral Raster Plot', fontsize=24)
        plt.xlabel("Time (ms)", fontsize=24)
        # ax.set_yticks([0,1,2,3,4,5,6,7,8])
        # ax.set_yticklabels(['inactive_press', 'active_press', 'reward_left', 'reward_right', 'lick_left', 'lick_right', 'ContextA', 'ContextB', 'ContextC']) #for some reason, I need to include +1 values here for the raster labels to display correctly
        ax.set_yticks([0,1,2,3,4])
        # ax.set_yticklabels(['active_press', 'reward_left', 'reward_right', 'lick_left', 'lick_right']) #for some reason, I need to include +1 values here for the raster labels to display correctly
        ax.set_yticklabels(['Lever press', 'Left reward', 'Right reward', 'Left lick', 'Right Lick'], fontsize=20) #for some reason, I need to include +1 values here for the raster labels to display correctly

        # flattens and sorts the All_Context_timestamps_list
        # All_Context_timestamps_list_flattened = [item for sublist in All_Context_timestamps_list for item in sublist]
        # All_Context_timestamps_list_flattened_and_sorted = sorted(All_Context_timestamps_list_flattened)
        # All_Context_timestamps_list_flattened_and_sorted.append(3600000)  #
        # inactive_press_list_by_Context = dict(
        #     pd.cut(inactive_press_df['Time'], bins=All_Context_timestamps_list_flattened_and_sorted,
        #            labels=Context_changes, ordered=False).value_counts(
        #         sort=False))  # bin the times in num_of_bins for eg, licks
        # keys_list = list(
        #     inactive_press_list_by_Context.keys())  # extracts the keys from an example dictionary, and creates a list from them
        #
        # start_block = 0
        # end_block = 60
        # for i in range(start_block, end_block):
        #     if keys_list[i].startswith('ContextA'):
        #         plt.axvspan(i - 0.5, i + 0.5, color='cyan', alpha=0.2)
        #     if keys_list[i].startswith('ContextB'):
        #         # plt.axvspan(i-0.5,i+0.5, color='thistle',alpha=0.3)
        #         plt.axvspan(i - 0.5, i + 0.5, color='darkgreen', alpha=0.2)
        #     if keys_list[i].startswith('ContextC'):
        #         plt.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.01)
        #
        plt.tight_layout()
        plt.show()
        # format='png'
        # plt.savefig('../figures/AB_preliminary_raster.' + format, format=format, dpi=300)
        
    parent = Tk() #instantiation of the Tk class from the tkinter library
    def get_start_time(): 
        start_time_sec = slider.get() #sets start_time_sec to the current slider value
        time_list.append(start_time_sec) #appends start_time_sec to this list; time_list used for time binning appropriately
        label.config(text = 'start time set: ' + str(start_time_sec)) #prints this in the GUI after the button is pressed
    def get_end_time():
        end_time_sec = slider.get() #sets end_time_sec to the current slider value
        time_list.append(end_time_sec) #appends end_time_sec to this list
        label.config(text = 'end time set: ' + str(end_time_sec)) #prints this in the GUI after the button is pressed
    
    parent.title('raster plot configuration') #title atop the GUI window
    slider = Scale(parent, from_=0, to=3600, orient=HORIZONTAL,label='raster_plot_seconds',tickinterval=500,
            sliderlength=15,length=300,bg='white',resolution=50) #parent: instantiation of the Tk class; from_: start value of slider; to: end value of slider; orient: Vert or Horiz; label: appears above the slider; resolution: 0-50-100-150 etc are the possible slider values 
    Button(parent, text='start_time', command=get_start_time).pack() #button calls get_start_time each time it is pressed
    Button(parent, text='end_time', command=get_end_time).pack() #button calls get_end_time each time it is pressed
    Button(parent, text='create_new_plot', command=new_plot).pack() #button calls new_plot each time it is pressed
    label = Label(parent) #permits the use of label.config which prints a string to the GUI
    label.pack() #pack organizes each component of the GUI
    slider.pack() #pack organizes each component of the GUI
    mainloop() #initiates the GUI
    
def generate_context_daily_line_plot(name_of_cleaned_csv_file):    
    # get_ipython().run_line_magic('matplotlib', 'inline')
    cleaned_csv = pd.read_csv(name_of_cleaned_csv_file) #opens csv
    
    #for general purpose binning; can reuse this elsewhere
    cleaned_csv = cleaned_csv.loc[(cleaned_csv['Output'] != 'session_begins_here')] #deletes 'session_begins_here' from top of dataframe
    
    #creates sub-dfs for each output and for each context change
    inactive_press_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'inactive_press')]
    active_press_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'active_press')]
    reward_left_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'reward_left')]
    reward_right_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'reward_right')]
    lick_left_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'lick_left')]
    lick_right_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'lick_right')]
    ContextA_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'ContextA')]
    ContextB_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'ContextB')]
    ContextC_df = cleaned_csv.loc[(cleaned_csv['Output'] == 'ContextC')]  
    
    #creates timestamp lists for each Context change
    ContextA_timestamp_list = ContextA_df.iloc[:,0].to_list() 
    ContextB_timestamp_list = ContextB_df.iloc[:,0].to_list()
    ContextC_timestamp_list = ContextC_df.iloc[:,0].to_list()
    
    #appends each timestamp list to the All_Context_timestamps_list
    All_Context_timestamps_list = []
    All_Context_timestamps_list.append(ContextA_timestamp_list)
    All_Context_timestamps_list.append(ContextB_timestamp_list)
    All_Context_timestamps_list.append(ContextC_timestamp_list)
    
    #flattens and sorts the All_Context_timestamps_list
    All_Context_timestamps_list_flattened = [item for sublist in All_Context_timestamps_list for item in sublist]
    All_Context_timestamps_list_flattened_and_sorted = sorted(All_Context_timestamps_list_flattened)
    All_Context_timestamps_list_flattened_and_sorted.append(3600000) #
    #creates a double-ended queue (a list that's good for popping elements from both ends)
    ContextA_list, ContextB_list, ContextC_list = deque([]), deque([]), deque([])
    Context_changes = []
    
    #for loops are for creating separate lists for the names of each context in the correct numbers 
    #(20:ContextA, 20:ContextB, 40:ContextC)
    #followed by ordering the properly with corresponding numbers (eg, ContextB1, ContextC1, ContextA1, ContextC2, etc.)
    for i in range(20): 
        ContextA_list.append('ContextA' + str(i+1)) #appends ContextA1, ContextA2 etc into list up to ContextA20
        ContextB_list.append('ContextB' + str(i+1)) #appends ContextB1, ContextB2 etc into list up to ContextB20
    for i in range(40):
        ContextC_list.append('ContextC' + str(i+1)) #appends ContextC1, ContextC2 etc into list up to ContextC40
    
    for i in range(20): #in Context_changes list, order the Contexts as they appear to the animal (ContextB1, ContextC1, ContextA1, ContextC2 etc.)
        Context_changes.append(ContextB_list[0])
        ContextB_list.popleft()
        Context_changes.append(ContextC_list[0])
        ContextC_list.popleft()
        Context_changes.append(ContextA_list[0])
        ContextA_list.popleft()
        Context_changes.append(ContextC_list[0])
        ContextC_list.popleft()
#     Context_changes.pop() #needs to pop the final ContextC since the session ends there

    #creates dictionaries with the Context keys (eg, ContextB4) coupled with the associated action counts in each of those blocks
    #longer than it needs to be, but dictionaries help to keep the values coupled with the keys for troubleshooting
    inactive_press_list_by_Context = dict(pd.cut(inactive_press_df['Time'], bins=All_Context_timestamps_list_flattened_and_sorted, 
                                      labels = Context_changes ,ordered=False).value_counts(sort=False)) #bin the times in num_of_bins for eg, licks
    active_press_list_by_Context = dict(pd.cut(active_press_df['Time'], bins=All_Context_timestamps_list_flattened_and_sorted, 
                                      labels = Context_changes ,ordered=False).value_counts(sort=False)) #bin the times in num_of_bins for eg, licks
    reward_left_list_by_Context = dict(pd.cut(reward_left_df['Time'], bins=All_Context_timestamps_list_flattened_and_sorted, 
                                      labels = Context_changes ,ordered=False).value_counts(sort=False)) #bin the times in num_of_bins for eg, licks
    reward_right_list_by_Context = dict(pd.cut(reward_right_df['Time'], bins=All_Context_timestamps_list_flattened_and_sorted, 
                                      labels = Context_changes ,ordered=False).value_counts(sort=False)) #bin the times in num_of_bins for eg, licks
    lick_left_list_by_Context = dict(pd.cut(lick_left_df['Time'], bins=All_Context_timestamps_list_flattened_and_sorted, 
                                      labels = Context_changes ,ordered=False).value_counts(sort=False)) #bin the times in num_of_bins for eg, licks
    lick_right_list_by_Context = dict(pd.cut(lick_right_df['Time'], bins=All_Context_timestamps_list_flattened_and_sorted, 
                                      labels = Context_changes ,ordered=False).value_counts(sort=False)) #bin the times in num_of_bins for eg, licks
    
    #gathers keys and values for each output
    inactive_press_keys = list(inactive_press_list_by_Context.keys())
    inactive_press_values = list(inactive_press_list_by_Context.values())
    
    active_press_keys = list(active_press_list_by_Context.keys())
    active_press_values = list(active_press_list_by_Context.values())
    
    reward_left_keys = list(reward_left_list_by_Context.keys())
    reward_left_values = list(reward_left_list_by_Context.values())
    
    reward_right_keys = list(reward_right_list_by_Context.keys())
    reward_right_values = list(reward_right_list_by_Context.values())
    
    lick_left_keys = list(lick_left_list_by_Context.keys())
    lick_left_values = list(lick_left_list_by_Context.values())
    
    lick_right_keys = list(lick_right_list_by_Context.keys())
    lick_right_values = list(lick_right_list_by_Context.values())
    
    list_to_generate_per_minute_calculation = [inactive_press_values,active_press_values,reward_left_values,reward_right_values,lick_left_values,lick_right_values]
    
    for i in range(len(list_to_generate_per_minute_calculation)):
        j = 1
        for p in range(len(list_to_generate_per_minute_calculation[i])):
            try:
                temp_num = list_to_generate_per_minute_calculation[i][j]*2
                list_to_generate_per_minute_calculation[i][j] = temp_num
                j+=2
            except:
                pass
        
    keys_list = list(inactive_press_list_by_Context.keys()) #extracts the keys from an example dictionary, and creates a list from them

    # start_block = int(input('Select starting Context (integer between 0 and 80): '))
    # end_block = int(input('Select ending Context (integer between 0 and 80): '))
    start_block=0
    # end_block=60
    end_block=30

    #plots below here
    #first plot is for active_presses, inactive_presses, and rewards
    # fig = plt.figure(figsize=(18,8))
    # plt.plot(range(start_block,end_block), inactive_press_values[start_block:end_block], color="lightgray",linewidth=2.5) #start_block and end_block are parameters;
    # plt.plot(range(start_block,end_block), active_press_values[start_block:end_block], color="black",linewidth=2.5) #start_block is the first context selected (eg, 0-->ContextB1)
    # plt.legend(['inactive_press', 'active_press'],prop={'size': 20})
    # plt.xlabel('Context Block', fontsize=24)
    # plt.ylabel('Counts/min',fontsize=24)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # for i in range(start_block,end_block): #this creates vertical shading for each context change, works well
    #     if keys_list[i].startswith('ContextA'):
    #         plt.axvspan(i-0.5,i+0.5, color='cyan',alpha=0.08)
    #     if keys_list[i].startswith('ContextB'):
    #         plt.axvspan(i-0.5,i+0.5, color='thistle',alpha=0.3)
    #     if keys_list[i].startswith('ContextC'):
    #         plt.axvspan(i-0.5,i+0.5, color='lightgray',alpha=0.01)
    # plt.show()

    #second plot is for rewards
    fig = plt.figure(figsize=(18,8))
    # plt.plot(range(start_block,end_block), reward_left_values[start_block:end_block], color='orchid',linewidth=2.5, label='Left reward') #end_block is the last context selected (eg, 4-->ContextC2)
    # plt.plot(range(start_block,end_block), reward_right_values[start_block:end_block], color="cyan",linewidth=2.5, label='Right reward') #start_block and end_block are parameters;
    # plt.legend(['reward_left','reward_right'],prop={'size': 20})
    # plt.xlabel('Context Block',fontsize=24)
    # plt.ylabel('Counts/min',fontsize=24)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # for i in range(start_block,end_block):
    #     if keys_list[i].startswith('ContextA'):
    #         plt.axvspan(i-0.5,i+0.5, color='cyan',alpha=0.2)
    #     if keys_list[i].startswith('ContextB'):
    #         plt.axvspan(i-0.5,i+0.5, color='darkgreen',alpha=0.2)
    #     if keys_list[i].startswith('ContextC'):
    #         plt.axvspan(i-0.5,i+0.5, color='lightgray',alpha=0.01)
    # plt.show()

    #third plot is for licks
    # fig = plt.figure(figsize=(18,8))
    plt.plot(range(start_block,end_block), lick_left_values[start_block:end_block], color='k',linewidth=2.5, label='Left lick')
    plt.plot(range(start_block,end_block), lick_right_values[start_block:end_block], color='gray',linewidth=2.5, label='Right lick', linestyle='--')
    plt.legend(fancybox=False, fontsize=28)
    # plt.legend(['lick_left','lick_right'],prop={'size': 20})
    plt.xlabel('Context Block',fontsize=24)
    plt.ylabel('Counts/min',fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for i in range(start_block,end_block):
        if keys_list[i].startswith('ContextA'):
            plt.axvspan(i-0.5,i+0.5, color='cyan',alpha=0.2)
        if keys_list[i].startswith('ContextB'):
            # plt.axvspan(i-0.5,i+0.5, color='thistle',alpha=0.3)
            plt.axvspan(i-0.5,i+0.5, color='darkgreen',alpha=0.2)
        if keys_list[i].startswith('ContextC'):
            plt.axvspan(i-0.5,i+0.5, color='lightgray',alpha=0.01)
    format = 'png'
    plt.xlim([-1,30])
    plt.title('Lick Choice Summary', fontsize=24)
    plt.tight_layout()
    plt.savefig('../figures/AB_preliminary_lineplot.' + format, format=format, dpi=300)
    # plt.show()
    
    #fourth plot is for active to inactive press ratio
    # fig = plt.figure(figsize=(18,8))
    # inactive_press_values = [inactive_press_values[i] + 1 for i in range(len(inactive_press_values))]
    # active_press_values = [active_press_values[i] + 1 for i in range(len(active_press_values))]
    # acitve_to_inactive_press_ratio = [active_press_values[i]/inactive_press_values[i] for i in range(len(active_press_values))]
    # plt.plot(range(start_block,end_block), acitve_to_inactive_press_ratio[start_block:end_block], color="black",linewidth=2.5)
    # plt.legend(['active_to_inactive_press_ratio'],prop={'size': 20})
    # plt.xlabel('Context Block',fontsize=24)
    # plt.ylabel('Active to Inactive Press Ratio',fontsize=24)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # for i in range(start_block,end_block):
    #     if keys_list[i].startswith('ContextA'):
    #         plt.axvspan(i-0.5,i+0.5, color='cyan',alpha=0.08)
    #     if keys_list[i].startswith('ContextB'):
    #         plt.axvspan(i-0.5,i+0.5, color='thistle',alpha=0.3)
    #     if keys_list[i].startswith('ContextC'):
    #         plt.axvspan(i-0.5,i+0.5, color='lightgray',alpha=0.01)
    # plt.show()
    
    # fifth plot is for active to inactive press ratio
#     fig = plt.figure(figsize=(18,8))
#     lick_left_values = [lick_left_values[i] + 1 for i in range(len(lick_left_values))]
#     lick_right_values = [lick_right_values[i] + 1 for i in range(len(lick_right_values))]
#     left_to_right_lick_ratio = [lick_left_values[i]/lick_right_values[i] for i in range(len(lick_left_values))]
#     plt.plot(range(start_block,end_block), left_to_right_lick_ratio[start_block:end_block], color="black",linewidth=2.5)
# #     plt.legend(['lick_left_to_right_ratio'],prop={'size': 20})
#     plt.xlabel('Context Block',fontsize=24)
#     plt.ylabel('Lick Left to Right Ratio',fontsize=24)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     for i in range(start_block,end_block):
#         if keys_list[i].startswith('ContextA'):
#             plt.axvspan(i-0.5,i+0.5, color='cyan',alpha=0.08)
#         if keys_list[i].startswith('ContextB'):
#             plt.axvspan(i-0.5,i+0.5, color='thistle',alpha=0.3)
#         if keys_list[i].startswith('ContextC'):
#             plt.axvspan(i-0.5,i+0.5, color='lightgray',alpha=0.01)
#     plt.show()
    
# generate_daily_raster('CleanedMF02_2023-05-01_150026.log.csv')
generate_context_daily_line_plot('CleanedMF02_2023-05-01_150026.log.csv')

