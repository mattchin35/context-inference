import pandas as pd
import os 
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML
pd.set_option('display.max_rows', None) #these four lines permit printing of the entire data frame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
display(HTML("<style>.container { width:100% !important; }</style>"))

def process_file(directory, file_path):    
    os.chdir(directory)
    
    # Read the file into a DataFrame, skipping the first row
    df = pd.read_csv(file_path, sep=';', skiprows=1, on_bad_lines='skip', usecols=[1, 3])
    
    # Rename the columns
    df.columns = ['Time', 'Output']
    
    # Subtract 'exit_standby' time value from every element
    exit_standby_time = df.loc[df['Output'] == 'exit_standby', 'Time'].values[0]
    df['Time'] = df['Time'] - exit_standby_time
    
    # Remove rows with negative 'Time' values
    df = df[df['Time'] >= 0]

    # You can use the str.replace() function
    df = df.replace({'Output': r'.*ITI.*'}, {'Output': 'ITI'}, regex=True)
    
    # Get unique values from the 2nd column
    unique_values = df['Output'].unique().tolist()
    
    # Subset the DataFrame based on the elements of the 2nd column
    subsets = {}
    for value in unique_values:
        subsets[value] = df[df['Output'] == value]
    
    # Extract specific keys from the subsets dictionary
    specific_keys = ['exit_standby', 'left_entry', 'right_entry', 'pump1_reward_0', 'pump1_reward_3',
                     'pump2_reward_0', 'pump2_reward_3', 'ITI', 
                     'enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval',
                     'enter_ContextC1', 'enter_ContextC2']
    
    # Filter the subsets dictionary to include only specific keys
    specific_subsets = {key: subsets[key] for key in specific_keys if key in subsets}
    
    # Save the specific subsets dictionary to a new DataFrame
    df_new = pd.concat(specific_subsets.values())
    
    # Save the new DataFrame to a CSV file
    new_file_path = os.path.join(directory, 'cleaned_' + os.path.basename(file_path))
    
    if os.path.exists(new_file_path):
        print('file_already_exists')
    else:
        df_new.to_csv(new_file_path, index=False)
        print('cleaned_file_created')

    return specific_subsets

#update according to date of interest############################
current_date = '2023-06-13'

# directories = ["/home/lukelab/Desktop/liquid_SA_data/for_matt/for_matt_wip/"]
directories = ["./"]

results = {'MF06': []}

# Save the original working directory
original_directory = os.getcwd()

# Iterate through directories
for directory in directories:
    # Reset the working directory at the start of each iteration
    os.chdir(original_directory)
    
    # Traverse directory, and for each file
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # If it's a log file
            if filename.endswith('.log') and current_date in filename and not 'cleaned' in filename:
                print(f'Processing file: {filename}')
                result = process_file(dirpath, filename)
                prefix = filename[:4]  # Get the first four characters of the filename
                if prefix in results:
                    results[prefix].append(result)

for key, value in results.items():
    print(f'For {key}, processed {len(value)} files.')
    


#####################RASTER PLOT############################
def generate_daily_raster(specific_subsets): 
    # get_ipython().run_line_magic('matplotlib', 'inline')
    
    fig = None # Initialize fig as None in the outer scope
    
    # Concatenate the dataframes in the specific_subsets dictionary into a single dataframe
    cleaned_df = pd.concat(specific_subsets.values())
    cleaned_df = cleaned_df[cleaned_df['Output'] != 'exit_standby']
    
    # output_list = ['left_entry', 'right_entry', 'pump1_reward_0', 'pump1_reward_3', 'pump2_reward_0',
    #                'pump2_reward_3', 'ITI', 'enter_intercontext_interval',
    #                'enter_ContextC1', 'enter_ContextC2', 'enter_ContextA', 'enter_ContextB']

    output_list = ['left_entry', 'right_entry',
                   'pump1_reward_3', 'pump2_reward_3',
                   'enter_ContextA', 'enter_ContextB']

    arrays_dict = {}

    for output in output_list:
        sub_df = cleaned_df[cleaned_df['Output'] == output]

        if output in ['enter_ContextA', 'enter_ContextB', 'enter_ContextC1', 'enter_ContextC2'] and sub_df.empty:
            continue

        arrays_dict[output + '_timestamp_array'] = np.array(sub_df['Time'])                      

    time_list =[]
        
    def new_plot(): 
        nonlocal fig # Make fig a nonlocal variable

        abridged_arrays_dict = {}

        for key, value in arrays_dict.items():
            abridged_array = value[(value >= time_list[0]) & (value <= time_list[1])]
            abridged_arrays_dict[key + '_abridged'] = abridged_array

        array_of_timestamp_arrays = np.array(list(abridged_arrays_dict.values()),dtype=object)
        label.config(text = 'raster plot created')
        time_list.clear()
        
        if 'enter_ContextA' in cleaned_df['Output'].values:
            # yticklabels = ['left_entry', 'right_entry', 'left_reward_0', 'left_reward_3', 'right_reward_0',
            #             'right_reward_3', 'ITI',
            #             'enter_intercontext_interval', 'enter_ContextA', 'enter_ContextB']
            yticklabels = ['Left lick', 'Right lick',
                           'Left reward', 'Right reward',
                           'Context A', 'Context B']
        else:
            yticklabels = ['Left lick', 'Right lick', 'left_reward_0', 'left_reward_3', 'right_reward_0',
                        'right_reward_3', 'ITI',
                        'enter_intercontext_interval', 'enter_ContextC1', 'enter_ContextC2']
            
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(10)
        fig.set_figwidth(18)
        ax.eventplot(array_of_timestamp_arrays, linelengths=0.6, linewidths=0.15, color='black')
        ax.set_title('Behavioral Raster Plot', fontsize=24)
        plt.xlabel("Time (seconds)", fontsize=24)
        # ax.set_yticks([0,1,2,3,4,5,6,7,8,9])
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=20)
        plt.xticks(fontsize=20)
        plt.tight_layout()
        plt.show()

    def save_plot():
        nonlocal fig # Use the nonlocal fig variable
        if fig is not None:
            fig.savefig('WIP_raster_plot.svg', format='svg')
            label.config(text = 'Plot saved as SVG')
        else:
            label.config(text = 'No plot to save yet. Create a new plot first.')
    
    def get_start_time(): 
        start_time_sec = slider.get() #sets start_time_sec to the current slider value
        time_list.append(start_time_sec) #appends start_time_sec to this list; time_list used for time binning appropriately
        label.config(text='start time set: ' + str(start_time_sec)) #prints this in the GUI after the button is pressed
    def get_end_time():
        end_time_sec = slider.get() #sets end_time_sec to the current slider value
        time_list.append(end_time_sec) #appends end_time_sec to this list
        label.config(text='end time set: ' + str(end_time_sec)) #prints this in the GUI after the button is pressed
    
    parent = Tk() #instantiation of the Tk class from the tkinter library
    parent.title('raster plot configuration') #title atop the GUI window
    slider = Scale(parent, from_=0, to=3600, orient=HORIZONTAL,label='raster_plot_seconds',tickinterval=500,
            sliderlength=15,length=300,bg='white',resolution=20) #parent: instantiation of the Tk class; from_: start value of slider; to: end value of slider; orient: Vert or Horiz; label: appears above the slider; resolution: 0-50-100-150 etc are the possible slider values 
    Button(parent, text='start_time', command=get_start_time).pack() #button calls get_start_time each time it is pressed
    Button(parent, text='end_time', command=get_end_time).pack() #button calls get_end_time each time it is pressed
    Button(parent, text='create_new_plot', command=new_plot).pack() #button calls new_plot each time it is pressed
    Button(parent, text='save_plot', command=save_plot).pack() #button calls save_plot each time it is pressed
    label = Label(parent) #permits the use of label.config which prints a string to the GUI
    label.pack() #pack organizes each component of the GUI
    slider.pack() #pack organizes each component of the GUI
    mainloop() #initiates the GUI

# Usage:
current_mouse = 'MF06'
# generate_daily_raster(results[current_mouse][0])





####################BINNING#########################
output_list = ['left_entry', 'right_entry', 'pump1_reward_1', 'pump1_reward_3',
               'pump2_reward_1', 'pump2_reward_3', 'current_ITI_2', 'current_ITI_3', 'current_ITI_4']

transition_list = ['enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval', 'enter_ContextC1', 'enter_ContextC2']

# Iterate over the main dictionary
for main_key in results.keys():

    # Get the sub-dictionary
    sub_dict = results[main_key][0]

    # Generate bins based on the timestamps of the transition list
    bins = []
    for transition in transition_list:
        if transition in sub_dict:
            bins.extend(sub_dict[transition]['Time'].values)
    bins = np.array(bins)
    bins = np.sort(bins)  # make sure the bins are in ascending order

    # Iterate over the output list
    for output in output_list:
        if output in sub_dict:
            # Bin the 'Time' values of the output
            sub_dict[output]['Bin'] = pd.cut(sub_dict[output]['Time'], bins, include_lowest=True)

            # Count the number of 'Time' values in each bin
            counts = sub_dict[output].groupby('Bin').count()



########################PLOT BINS########################
# output_list = ['left_entry', 'right_entry', 'pump1_reward_1', 'pump1_reward_3',
#                'pump2_reward_1', 'pump2_reward_3', 'current_ITI_2', 'current_ITI_3', 'current_ITI_4']

output_list = ['left_entry', 'right_entry']
transition_list = ['enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval']

# colors = ['red', 'green', 'red', 'purple', 'orange']  # Add more colors if needed
colors = ['cyan', 'darkgreen', 'lightgray', 'purple', 'lightgray']  # Add more colors if needed

# print('start_bin: ')
# start_bin = int(input())
# print('end_bin: ')
# end_bin = int(input())
# bin_range = [start_bin,end_bin]
bin_range = [0, 30]

# print('Do you want to save plots? (yes/no): ')
# save_plots = input().strip().lower()

for main_key in results.keys():
    sub_dict = results[main_key][0]

    bins = []
    bin_transitions = []
    for transition in transition_list:
        if transition in sub_dict:
            bins.extend(sub_dict[transition]['Time'].values)
            bin_transitions.extend([transition] * len(sub_dict[transition]['Time'].values))

    bins = np.array(bins)
    bin_transitions = np.array(bin_transitions)
    bins_sort_indices = np.argsort(bins)
    bins = bins[bins_sort_indices]
    bin_transitions = bin_transitions[bins_sort_indices]

    cmap = {transition: color for transition, color in zip(transition_list, colors)}
    fig, ax = plt.subplots(figsize=(12, 6))

    patches = []
    for transition in transition_list:
        color = cmap[transition]
        patch = mpatches.Patch(color=color, alpha=0.2, label=transition)
        patches.append(patch)

    plt_options = (('Left lick', '-', 'k'), ('Right lick', '--', 'gray'))
    for j, output in enumerate(output_list):
        if output in sub_dict:
            sub_dict[output]['Bin'] = pd.cut(sub_dict[output]['Time'], bins, include_lowest=True)
            counts = sub_dict[output].groupby('Bin').count()
            # fig, ax = plt.subplots(figsize=(12, 6))

            bin_centers = []
            count_values = []
            for i in range(len(bins) - 1):
                    ax.axvspan(bins[i], bins[i + 1], facecolor=cmap[bin_transitions[i]], alpha=0.075)
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
                    count_values.append(counts['Time'].iloc[i])

            # Plot line without markers
            ax.plot(bin_centers, count_values, linewidth=1, label=plt_options[j][0], linestyle=plt_options[j][1], color=plt_options[j][2])

            # plt.xlim(bin_centers[0], bin_centers[-1])
            # plt.ylim(bottom=0)

            plt.title(f"Lick Choice Summary", fontsize=16)
            plt.xlabel('Time(seconds)', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            # if save_plots == 'yes':
            # plt.show()

    plt.xlim(bin_centers[0], bin_centers[-1])
    plt.ylim(bottom=0)
    plt.legend(fancybox=False, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"WIP-block-actions.svg", format='svg')
