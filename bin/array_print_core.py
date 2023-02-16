#!/usr/bin/env python3

import os
from unittest import skip

import numpy as np
import pandas as pd
import datetime
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 
def csv_to_df(library_csv):
    csv_filepath = library_csv._selected_path + '/' + library_csv._selected_filename
    library_df = pd.read_csv(csv_filepath)
    column_names = list(library_df.columns)
    display(library_df)
    library_members = library_df['plate_position'].to_list()

    return library_df, column_names, library_members




def count_replicates(library_df, total_columns, total_rows, empty_columns, skip_rows):
    library_members = set(library_df['plate_position'])
    library_size = len(library_members)

    if skip_rows == 'n':
        rows = total_rows
    elif skip_rows == 'y':
        rows = total_rows/2
    columns = total_columns

    replicates = int((rows * columns)/(library_size))
    skip_rows = skip_rows.lower()

    if (empty_columns != 0) and (skip_rows == 'n'):
        print('Library contains', library_size, 'members. Accounting for skipped columns, the script will array', int(replicates), 'replicates per library member.')
    elif (empty_columns != 0) and (skip_rows == 'y'):
        print('Library contains', library_size, 'members. Accounting for skipped rows and columns, the script will array', int(replicates), 'replicates per library member.')
    elif skip_rows == 'n':
        print('Library contains', library_size, 'members. Will array', int(replicates), 'replicates per library member.')
    elif skip_rows == 'y':
        print('Library contains', library_size, 'members. Accounting for skipped rows, the script will array', int(replicates), 'replicates per library member.')
    else:
        print("Something looks wrong. Check your print presets.")

    return library_members, library_size, columns, rows, replicates, skip_rows




def generate_array(filename, library_df, total_columns, total_rows, skip_rows, column_names, user_binning, replicates, columns, rows, empty_columns):

    total_wells = (columns * rows)

    plate_number = column_names[0]
    plate_position = column_names[1]


    # ============
    ### Generate Array
    # Without catalytic binning
    if user_binning == 'no':
        zipped_list = zip(library_df[plate_number].to_list(), library_df[plate_position].to_list())
        muts_list = [str(x[0]) + x[1] for x in zipped_list]

        full_list = muts_list * replicates

        # Fill unclaimed wells
        remaining_chambers = total_wells - len(full_list)

        for i in range(int(remaining_chambers)):
            full_list.append(np.random.choice(muts_list))

        # Shuffle order of list
        shuffle(full_list)

    # With catalytic binning
    elif user_binning == 'yes':
        zipped_list = zip(library_df[plate_number].to_list(), library_df[plate_position].to_list())
        muts_list = [ str(x[0]) + x[1] for x in zipped_list ]

        # Generate lookup dictionary
        library_df['concat_plate_position'] = library_df[['plate_number', 'plate_position']].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
        catalytic_dict = pd.Series(library_df.catalytic_bin.values,index=library_df.concat_plate_position).to_dict()
        
        # Reverse the order of the lookup dictionary e.g. (k,v -> v,k)
        t = [ (catalytic_dict[i], i) for i in muts_list]

        d = defaultdict(list)
        for k, v in t:
            d[k].append(v)

        d = { k:list(v) for k, v in d.items() }

        full_list = []

        current_bin_list = []

        # Iterate through list of bin values
        for i in set(library_df.catalytic_bin.values):
            
            # Generate desired replicate, shuffle order of list
            current_bin_list = d[i] * replicates
            shuffle(current_bin_list)

            full_list = full_list + current_bin_list

            current_bin_list = []

        # Fill unclaimed wells
        remaining_chambers = total_wells - len(full_list)
        print('total wells are', total_wells)

        
    # Trim size of full list
    if remaining_chambers > 0: # Add items if not enough in list
        print('current length of list is', len(full_list))
        print("Adding " + str(remaining_chambers) + " replicates to fill array.")
        for i in range(int(remaining_chambers)):
            full_list.append(np.random.choice(muts_list))

    elif remaining_chambers < 0: # Remove items if too many in list
        print('current length of list is', len(full_list))
        print("Removing " + str(abs(remaining_chambers)) + " replicates to meet array size.")

        for i in range(int(abs(remaining_chambers))):
            choice = np.random.randint(0,len(full_list))
            full_list.pop(choice) # remove one from end of list


    ### Format Array ###

    # Convert list to df with dimensions of print
    print_array = np.array(full_list)
    if skip_rows == 'y':
        print_array = print_array.reshape(int(total_rows/2), total_columns)
    elif skip_rows == 'n':
        print_array = print_array.reshape(int(total_rows), total_columns)

    # Insert NA for row skips
    if skip_rows == 'y':
        for i in range(1, total_rows, 2):
            print_array = np.insert(print_array, i, np.nan, axis=0)

    # # Insert NA for column skips
    # if empty_columns != 0:
    #     empty_col_idxs = []

    #     for n in range(empty_columns):
    #         empty_col_idxs.append(np.random.randint(0,total_columns)) # non-inclusive

    #     for i in empty_col_idxs:
    #         print_array[:, int(i)] = np.nan(total_rows)


    print_df = pd.DataFrame(print_array, columns = list(range(total_columns)))

    # Save array
    cwd = os.getcwd()
    time = datetime.datetime.now()
    time = "_" + str(time).replace(" ", "_")

    project_path = cwd + '/' + filename + time
    try:
        os.mkdir(project_path)
    except OSError as error: 
        print(error)

    pd.DataFrame(print_df).to_csv(project_path + '/' + filename + '_array' + time + '.csv')

    if user_binning == 'n':
        catalytic_dict = []

    # Sum library member counts
    counts = print_df.apply(pd.value_counts, dropna=True)
    counts['Replicate count'] = counts.sum(axis=1)
    counts = pd.DataFrame(counts['Replicate count'])
    counts['Member'] = counts.index
    counts = counts.sort_values('Replicate count')
    counts = counts[counts['Member'] != 'nan']

    # return results
    if user_binning == 'yes':
        return print_df, project_path, time, counts, catalytic_dict
    else:
        return print_df, project_path, time, counts




def plot_metrics(library_members, print_df, library_df, replicates, counts, user_binning, filename, time, catalytic_dict):
    ### Initialize subplot ###
    sns.set(rc = {'figure.figsize':(30, 14)})
    sns.set_style('white')
    sns.set_context('poster')

    if user_binning == 'yes':
        fig, (ax2, ax1, ax3) = plt.subplots(1, 3, constrained_layout=True)
    else:
        fig, (ax2, ax1) = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle('Array Metrics for ' + filename + time)


    # ==================
    ### Plot member position ###
    value_to_int = {j:i for i,j in enumerate(pd.unique(print_df.values.ravel()))}
    n = len(library_df)

    # discrete colormap (n samples from a given cmap)
    cmap = sns.color_palette("plasma", n)
    ax1.set_facecolor("black") # Sets nan values to black
    categorical_heatmap_df = print_df.copy()

    ax1.set_title("Library Member Distriution")

    # Black frame
    for spine in ax1.spines.values():
        spine.set_visible(True)

    # Account for large libraries
    if len(library_members) < 80:
        sns.heatmap(categorical_heatmap_df.replace(value_to_int), cmap=cmap, square=True, linewidth=0.1, linecolor='black', ax=ax1).set_facecolor('white')
        colorbar = ax1.collections[0].colorbar 
        r = colorbar.vmax - colorbar.vmin 
        colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        colorbar.set_ticklabels(list(value_to_int.keys()))

    else:
        sns.heatmap(categorical_heatmap_df.replace(value_to_int), cmap=cmap, square=True, linewidth=0.1, linecolor='black', ax=ax1, cbar=False).set_facecolor('white')


    # ==================
    ### Plot counts barplot ###
    replicates = (1792/2)/len(counts['Member'])
    
    sns.color_palette("Spectral")
    sns.barplot(x='Member', y='Replicate count', data=counts, ax=ax2)
    ax2.set(xticks=[])
    ax2.set_title("Library Member Distribution")
    ax2.hlines(y = replicates, xmin = 0, xmax = len(library_members), ls='--', colors='black')
    ax2.set_ylim(0, max(counts['Replicate count']) * 2)

    low_replicates_df = counts[counts['Replicate count'] < replicates]

    textstr = '\n'.join([str(len(low_replicates_df)) + ' members were below the',
                        'replicate threshold.'
                        ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5, 
                fc=(0.8, 0.8, 1), ec=(0.5, 0.5, 1))

    # place a text box in upper left in axes coords
    ymin, ymax = ax2.get_ylim()
    ax2.text(1, # x position
            ymax * (3/4), # y position
            textstr,
            verticalalignment='top', bbox=props
            )


    # ==================
    ### Plot catalytic binning ###
    if user_binning == 'yes':

        catalytic_heatmap_df = print_df.copy()
        catalytic_heatmap_df = catalytic_heatmap_df.fillna(0)

        for i in catalytic_heatmap_df.columns:
            catalytic_heatmap_df[i] = catalytic_heatmap_df[i].map(catalytic_dict)

        # sns.color_palette("viridis", as_cmap=True)
        sns.heatmap(catalytic_heatmap_df, cmap='viridis', square=True, linewidth=0.1, linecolor='black', ax=ax3).set_facecolor('grey')
        ax3.set_title("Catalytic Bin Distribution")

        colorbar = ax3.collections[0].colorbar
    
    # ==================
    ### Save figure ###
    cwd = os.getcwd()
    plt.savefig(cwd + '/' + filename + time + '/' + filename + '_metrics.png')



def display_fld(print_df, total_columns, total_rows):
    for i in range(0, total_columns):
        for j in range(0, total_rows):
            current_fld_loc = str(i + 1) + '/' + str(j + 1) # add ones to change from 0-indexing
            current_array_val = print_df.iloc[i][j]

            # Insert blank for NaN values
            if type(current_array_val) != str:
                current_array_val = '\t'

            array_loc_print = current_array_val
            print(current_fld_loc + '\t' + array_loc_print + ',' + '\t' + '1,')



def write_fld(project_path, filename, print_df, total_columns, total_rows):
    time = datetime.datetime.now()
    time = "_" + str(time).replace(" ", "_")

    with open(project_path + '/' + filename + '_fld' + time + '.txt', 'w') as f:
        for i in range(0, total_columns):
            for j in range(0, total_rows):
                current_fld_loc = str(i + 1) + '/' + str(j + 1) # add ones to change from 0-indexing
                current_array_val = print_df.iloc[i][j]

                # Insert blank for NaN values
                if type(current_array_val) != str:
                    current_array_val = '\t'

                array_loc_print = current_array_val
                f.write(current_fld_loc + '\t' + array_loc_print + ',' + '\t' + '1,' + '\n')