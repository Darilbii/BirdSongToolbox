""" Functions for creating Larger Epochs that contain the smaller labeled epochs"""

import numpy as np
import mne


def determine_chunks_for_epochs(times):
    """ Epochs the Raw Data into Long Chunks that contain the handlabeled epochs

    Parameters
    ----------
    times : array
        array of the absolute start of the labels

    Returns
    -------
    logged : array
        array of the automated motif labels to use as benchmarks for the long epochs
    ledger : list
        list of each automated motif that occurs within each Epoch
        [Chunk (Epoch)] -> [Motifs in Epoch (Chunk)]
    """
    times_sorted = np.argsort(times)  # Sort the Motif Starts in Sequential Order
    roster = list(times_sorted[1:])  # Make a roster of the motifs in sequential order excluding the first
    ledger = []
    logged = []
    focus = times_sorted[0]  # Select the first Start motif focus
    candidate = focus  # Select the First Candidate for the End of the Chunk
    ledger_focus = [focus]
    counter = 0

    while len(roster) > 0:
        competitor = roster[0]  # Select the Candidate for the End of the Chunk
        distance = (times[competitor] - times[candidate]) / 30000  # Distance in Seconds

        # The Last Motif for the Entire Recording
        if len(roster) == 1:
            if counter > 0:
                ledger_focus.extend([competitor])  # Add the competitor to this Chunk's Ledger
                logged.append([focus, competitor])  # Save the Start and End of the Successful Chunk Pair
                ledger.append(ledger_focus)  # Append the Chunk's Completed Ledger to the Overall Ledger
                roster.pop(0)  # Close out the While Loop
            else:
                logged.append([focus, None])  # First Special Case Lone Motif
                ledger.append(ledger_focus)  # Append the Chunk's Completed Ledger to the Overall Ledger
                logged.append([competitor, None])  # Second Special Case Lone Motif (Nothing After It)
                ledger_focus = [competitor]  # Create new Ledger for the Next Chunk
                ledger.append(ledger_focus)  # Append the Chunk's Completed Ledger to the Overall Ledger
                roster.pop(0)  # Close out the While Loop

        # From the Starting Motif to the Second to Last Motif
        elif distance < 2:
            ledger_focus.extend([competitor])  # Add the competitor to this Chunk's Ledger
            candidate = roster.pop(0)  # Update to the new candidate
            counter = counter + 1  # Count the Number of Motifs Contained in the Motif

        elif distance < 30:
            ledger_focus.extend([competitor])  # Add the competitor to this Chunk's Ledger
            candidate = roster.pop(0)  # Update to the new candidate
            counter = counter + 1  # Count the Number of Motifs Contained in the Motif

        else:
            if counter > 0:
                logged.append([focus, candidate])  # Save the Start and End of the Successful Chunk Pair
                focus = competitor  # Update the Focus to the competitor [New Starting Motif]
                roster.pop(0)  # Clear the roster for the New Competitor
                ledger.append(ledger_focus)  # Append the Chunk's Completed Ledger to the Overall Ledger
                ledger_focus = [focus]  # Create new Ledger for the Next Chunk
                candidate = focus  # Update the New Start Candidate and remove them from the Roster
                counter = 0
            else:
                logged.append([focus, None])  # Special Case Lone Motif
                focus = competitor  # Update the New Start Focus to the competitor
                roster.pop(0)  # Clear the Special Lone Motif from the roster
                ledger.append(ledger_focus)  # Append the Chunk's Completed Ledger to the Overall Ledger
                ledger_focus = [focus]  # Create new Ledger for the Next Chunk
                candidate = focus  # Update the New Start Candidate
                counter = 0

    return logged, ledger


def epoch_lfp_ds_data(kwd_file, kwe_data, chunks, kwik_data=None,  verbose: bool=False):
    """ Epochs Neural Data from the KWD File and converts it to µV, Low-Pass Filters and Downsamples to 1 KHz

        Parameters:
        -----------
        kwd_file: h5py.File
            KWD file imported using h5py library
        kwe_data: dict
            dictionary of the events in the KWE file
            Keys:
                'motif_st': [# of Motifs]
                'motif_rec_num': [# of Motifs]

        Returns:
        --------
        lfp: ndarray
            Multidimensional array of Neural Raw signal Recording
            (Motif Length in Samples  x  Num. of Channels  x  Num. of Motifs)
        buff_chunks: list
            list of Epoch data based on chunks parameter
        chunk_index: list
            list of the Absolute Start and End of the Epochs

        Notes:
        ------
            The raw data are saved as signed 16-bit integers, in the range -32768 to 32767. They don’t have a unit. To
        convert to microvolts, just  multiply by 0.195. This scales the data into the range ±6.390 mV,
        with 0.195 µV resolution (Intan chips have a ±5 mV input range).

        """
    # Get The LFP Data

    # Steps
    # Low Pass Filter All of the Data
    # Grab Data Chunks Using Chunk Info
    # *Seperate from the Song Data*
    # Decimate all of the Chunk Data
    # Save the Chunks into a list of arrays [Chunks] -> (channels, Samples)

    fs = 30000  # Sampling Rate
    lpf_buffer = 20 * fs  # 10 sec Buffer for the Lowpass Filter
    chunk_buffer = 30 * fs  # 30 sec Buffer for Epoching

    buff_chunks = []
    chunk_index = []

    for index, (start, end) in enumerate(chunks):
        if end is None:
            end = start

        epoch_start = kwe_data['motif_st'][start]  # Start Time of Epoch (Chunk) in its Specific Recording
        epoch_end = kwe_data['motif_st'][end]  # End Time of Epoch (Chunk) in its Specific Recording
        rec_num = kwe_data['motif_rec_num'][start]  # Recording Number Epoch (Chunk) Occurs During
        kwd_rec_raw_data = kwd_file['recordings'][str(rec_num)]['data']  # Raw Data for this Recording Number
        rec_start = kwik_data['recordingStarts'][kwe_data['motif_rec_num'][start]]  # Start Sample of Recording

        if verbose:
            # Print out info about motif
            print('On Motif ', (index + 1), '/', len(chunks))

        chunk_start = int(epoch_start - (chunk_buffer + lpf_buffer))
        chunk_end = int(epoch_end + chunk_buffer + lpf_buffer)
        chunk_index.append([int((rec_start + epoch_start) - chunk_buffer), int((rec_start + epoch_end) + chunk_buffer)])

        worst_case = 0  # Hacky Solution to keeping tabs of whether a worst case is occuring

        # Handle Edge Cases where the Epochs go Beyond their Particular Recording
        if chunk_start < 0 or chunk_end > kwd_rec_raw_data.shape[0]:  # If the Epoch goes Beyond its Starting Recording
            # Initiate a empty chunk array the size of the Current Epoch (Chunk)
            duration = chunk_end - chunk_start  # the Full Length of the Entire Epoch
            chunk_array = np.zeros((duration, kwd_rec_raw_data.shape[1] - 1))

            # Case 1: Chunk starts before the start of Rec
            if chunk_start < 0:
                # Worst Case 1: The Buffer goes beyond the start of entire recording
                if rec_num == 0:
                    reduced_buffer = lpf_buffer + chunk_start
                    if reduced_buffer < 0:
                        print('Not enough of a Buffer for the First Recording')
                        break
                    chunk_array = kwd_rec_raw_data[:chunk_end, :-1]
                    worst_case = 1
                    print('worst')
                else:
                    # Get the Prior Recordings Data
                    prior_kwd_rec_raw_data = kwd_file['recordings'][str(rec_num - 1)]['data']
                    # Stitch the starting samples with the prior Recording
                    chunk_array[:np.abs(chunk_start), :] = prior_kwd_rec_raw_data[chunk_start:, :-1]
                    # Stitch the ending samples with the current Recording
                    chunk_array[np.abs(chunk_start):, :] = kwd_rec_raw_data[:chunk_end, :-1]

                if verbose:
                    print(f'Special Case 1: Recording {rec_num-1} to Recording {rec_num}')

            # Case 2: Chunk ends after the end of the REc
            if chunk_end > kwd_rec_raw_data.shape[0]:
                # Get Ending of Epoch in the Next Recording
                relative_chunk_end = chunk_end - kwd_rec_raw_data.shape[0]

                # Worst Case 2: The Buffer goes beyond the end of entire recording
                if rec_num == int(max(kwd_file['recordings'].keys())):
                    reduced_buffer = lpf_buffer - relative_chunk_end
                    if reduced_buffer < 0:
                        print('Not enough of a Buffer for the Last Recording')
                        break
                    chunk_array = kwd_rec_raw_data[chunk_start:, :-1]
                    worst_case = 2

                else:
                    # Get the Next Recordings Data
                    next_kwd_rec_raw_data = kwd_file['recordings'][str(rec_num + 1)]['data']
                    # Stitch the starting samples with the prior Recording
                    chunk_array[:-relative_chunk_end, :] = kwd_rec_raw_data[chunk_start:, :-1]
                    # Stitch the ending samples with the current Recording
                    chunk_array[-relative_chunk_end:, :] = next_kwd_rec_raw_data[:relative_chunk_end, :-1]
                if verbose:
                    print(f'Special Case 2: Recording {rec_num} to Recording {rec_num+1}')

            chunk_array = np.transpose(chunk_array * .195)  # Make the Correct Shape for mne with 0.195 µV resolution

        # print(epoch_start, 'to', epoch_end)  # Recording Number Motif Occurs During
        else:
            chunk_array = np.transpose(kwd_rec_raw_data[chunk_start:chunk_end, :-1]) * .195  # 0.195 µV resolution
        chunk_filt = mne.filter.filter_data(chunk_array,
                                            sfreq=fs,
                                            l_freq=None,
                                            h_freq=400,
                                            verbose=False)
        if worst_case == 0:
            buff_chunks.append(chunk_filt[:, lpf_buffer:-lpf_buffer:30])  # Remove the LPF Buffer and Downsample to 1KHz
        elif worst_case == 1:
            buff_chunks.append(
                chunk_filt[:, reduced_buffer:-lpf_buffer:30])  # Remove the LPF Buffer and Downsample to 1KHz
        else:
            buff_chunks.append(
                chunk_filt[:, lpf_buffer:-reduced_buffer:30])  # Remove the LPF Buffer and Downsample to 1KHz
    return buff_chunks, chunk_index


def epoch_bpf_audio(kwd_file, kwe_data, chunks, kwik_data=None,  verbose: bool=False):
    """Chunk the Audio and Bandpass Filter to remove noise """
    # Temporary Conformation to allow comparisons between Old Filtered Audio and the New Filtered Audio

    fs = 30000  # Sampling Rate
    lpf_buffer = 20 * fs  # 10 sec Buffer for the Lowpass Filter
    chunk_buffer = 30 * fs  # 30 sec Buffer for Epoching

    buff_chunks = []
    chunk_index = []

    for index, (start, end) in enumerate(chunks):
        if end is None:
            end = start

        epoch_start = kwe_data['motif_st'][start]  # Start Time of Epoch (Chunk) in its Specific Recording
        epoch_end = kwe_data['motif_st'][end]  # End Time of Epoch (Chunk) in its Specific Recording
        rec_num = kwe_data['motif_rec_num'][start]  # Recording Number Epoch (Chunk) Occurs During
        kwd_rec_raw_data = kwd_file['recordings'][str(rec_num)]['data']  # Raw Data for this Recording Number
        rec_start = kwik_data['recordingStarts'][kwe_data['motif_rec_num'][start]]  # Start Sample of Recording

        if verbose:
            # Print out info about motif
            print('On Motif ', (index + 1), '/', len(chunks), 'Duration: ', )

        chunk_start = int(epoch_start - (chunk_buffer + lpf_buffer))
        chunk_end = int(epoch_end + chunk_buffer + lpf_buffer)
        chunk_index.append([int((rec_start + epoch_start) - chunk_buffer), int((rec_start + epoch_end) + chunk_buffer)])

        worst_case = 0  # Hacky Solution to keeping tabs of whether a worst case is occuring

        # print(epoch_start, 'to', epoch_end)  # Recording Number Motif Occurs During

        # Handle Edge Cases where the Epochs go Beyond their Particular Recording
        if chunk_start < 0 or chunk_end > kwd_rec_raw_data.shape[0]:  # If the Epoch goes Beyond its Starting Recording
            # Initiate a empty chunk array the size of the Current Epoch (Chunk)
            duration = chunk_end - chunk_start  # the Full Length of the Entire Epoch
            chunk_array = np.zeros(duration)

            # Case 1: Chunk starts before the start of Rec
            if chunk_start < 0:
                # Worst Case 1: The Buffer goes beyond the start of entire recording
                if rec_num == 0:
                    reduced_buffer = lpf_buffer + chunk_start
                    if reduced_buffer < 0:
                        print('Not enough of a Buffer for the First Recording')
                        break
                    chunk_array = kwd_rec_raw_data[:chunk_end, -1]
                    worst_case = 1
                    print('worst')
                else:
                    # Get the Prior Recordings Data
                    prior_kwd_rec_raw_data = kwd_file['recordings'][str(rec_num - 1)]['data']
                    # Stitch the starting samples with the prior Recording
                    chunk_array[:np.abs(chunk_start)] = prior_kwd_rec_raw_data[chunk_start:, -1]
                    # Stitch the ending samples with the current Recording
                    chunk_array[np.abs(chunk_start):] = kwd_rec_raw_data[:chunk_end, -1]

                if verbose:
                    print(f'Special Case 1: Recording {rec_num-1} to Recording {rec_num}')

            # Case 2: Chunk ends after the end of the REc
            if chunk_end > kwd_rec_raw_data.shape[0]:
                # Get Ending of Epoch in the Next Recording
                relative_chunk_end = chunk_end - kwd_rec_raw_data.shape[0]

                # Worst Case 2: The Buffer goes beyond the end of entire recording
                if rec_num == int(max(kwd_file['recordings'].keys())):
                    reduced_buffer = lpf_buffer - relative_chunk_end
                    if reduced_buffer < 0:
                        print('Not enough of a Buffer for the Last Recording')
                        break
                    chunk_array = kwd_rec_raw_data[chunk_start:, -1]
                    worst_case = 2
                else:
                    # Get the Next Recordings Data
                    next_kwd_rec_raw_data = kwd_file['recordings'][str(rec_num + 1)]['data']
                    # Stitch the starting samples with the prior Recording
                    chunk_array[:-relative_chunk_end] = kwd_rec_raw_data[chunk_start:, -1]
                    # Stitch the ending samples with the current Recording
                    chunk_array[-relative_chunk_end:] = next_kwd_rec_raw_data[:relative_chunk_end, -1]
                if verbose:
                    print(f'Special Case 2: Recording {rec_num} to Recording {rec_num+1}')

            chunk_array = chunk_array * .195  # Make the Correct Shape for mne with 0.195 µV resolution

        else:
            chunk_array = kwd_rec_raw_data[chunk_start:chunk_end, -1] * .195  # 0.195 µV resolution
        chunk_filt = mne.filter.filter_data(chunk_array, sfreq=fs, l_freq=300, h_freq=10000, fir_design='firwin2',
                                            verbose=False)
        if worst_case == 0:
            buff_chunks.append(chunk_filt[lpf_buffer:-lpf_buffer])  # Remove the LPF Buffer|Downsample to 1KHz

        elif worst_case == 1:
            buff_chunks.append(chunk_filt[reduced_buffer:-lpf_buffer])  # Remove the LPF Buffer|Downsample to 1KHz

        else:
            buff_chunks.append(chunk_filt[lpf_buffer:-reduced_buffer])  # Remove the LPF Buffer|Downsample to 1KHz

    return buff_chunks



