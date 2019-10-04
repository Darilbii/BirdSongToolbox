""" Functions for creating Larger Epochs that contain the smaller labeled epochs"""

import numpy as np

# from neurodsp import filt
# TODO: implement Version that utilizes neurodsp
try:
    import mne
    # from mne.filter import filter_data
    _has_mne = True
except ImportError:
    _has_mne = False


def determine_chunks_for_epochs(times, search_buffer=30):
    """ Epochs the Raw Data into Long Chunks that contain the handlabeled epochs (from .kwe)

    Parameters
    ----------
    times : array
        array of the absolute start of the labels
    search_buffer : int
        Buffer in seconds to group relatively close epochs

    Returns
    -------
    logged : list
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

        elif distance < search_buffer:
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


def get_chunk_from_kwd(start, end, chunk_buffer, lpf_buffer, kwd_file, kwe_data, index: list, verbose: bool = False):
    """ Gets one Epoch(Chunk) from the KWD File and returns Meta-Data on the Epoch for Pre-Processing

    Parameters
    ----------
    start : int
        Index for the first epoch(KWE) in the Epoch(Chunk)
    end : int
        Index for the last epoch(KWE) in the Epoch(Chunk)
    chunk_buffer : int
        Number of samples to buffer the chunk from the start of the first epoch(KWE) to the end of the last epoch(KWE),
         defaults to 30 secs
        chunk_buffer = 30 * fs  # 30 sec Buffer for Epoching
    lpf_buffer : int
        Number of samples to buffer the chunk itself for Low-pass filtering and downsampling, defaults to 20 secs
    kwd_file : h5py.File
        KWD file imported using h5py library
    kwe_data : dict
        dictionary of the events in the KWE file
        Keys:
            'motif_st': [# of Motifs]
            'motif_rec_num': [# of Motifs]
    index : list
        list of what recording channels to select (channels in terms of columns in kwd file)
    verbose : bool
        If True the Function prints out useful statements, defaults to False

    Returns
    -------
    chunk_array : array-like 1d or 2d
        The selected data for the Epoch)(Chunk)
    chunk_index : list
        list of the Absolute Start and End of the Epoch(Chunk) excluding the lpf_buffer
    worst_case : int or float
        The case_id of the Epoch
    reduced_buffer : int
        If the epoch is a edge case with a reduced Buffer then this will be a integer describing how much of the buffer
        is possible, else it is None

    Notes
    -----
    Case 0 : the Epoch(Chunk) and its buffers are within the entire Recording
    Case 1 : the starting buffer is clipped off (reduced)
    Case 1.1 : the entire Starting Buffer is gone
    Case 2 : the ending buffer is clipped off (reduced)
    Case 2.1 : the entire ending buffer is clipped off (reduced)

    """
    # TODO: Change the first two Parameters to be a Tuple to reduce space and reduncancy
    # start, end = chunk_comps
    epoch_start = kwe_data['motif_st'][start]  # Start Time of Epoch(Chunk) in its Specific Recording
    epoch_end = kwe_data['motif_st'][end]  # End Time of Epoch (Chunk) in its Specific Recording
    rec_num = kwe_data['motif_rec_num'][start]  # Recording Number Epoch(Chunk) Occurs During
    rec_num_end = kwe_data['motif_rec_num'][end]  # Recording Number End Epoch(Chunk) Occurs During
    kwd_rec_raw_data = kwd_file['recordings'][str(rec_num)]['data']  # Raw Data for this Recording Number

    rec_start = kwd_file['recordings'][str(rec_num)].attrs.get('start_sample')  # Start Sample of Rec (.attrs of hdf)

    chunk_start = int(epoch_start - (chunk_buffer + lpf_buffer))
    chunk_end = int(epoch_end + chunk_buffer + lpf_buffer)

    if rec_num_end > rec_num:
        if rec_num_end == rec_num + 1:
            chunk_end = chunk_end + kwd_rec_raw_data.shape[0]
        else:
            raise ValueError('The Chunk Occurs Across more than 2 Recording and there is no function for that yet')

    worst_case = 0  # Hack Solution to keeping tabs of whether a worst case has occurred

    # Handle Edge Cases where the Epochs go Beyond their Particular Recording
    if chunk_start < 0 or chunk_end > kwd_rec_raw_data.shape[0]:  # If the Epoch goes Beyond its Starting Recording
        # Initiate a empty chunk array the size of the Current Epoch (Chunk)
        duration = chunk_end - chunk_start  # the Full Length of the Entire Epoch
        chunk_array = np.zeros((duration, len(index)))

        # Case 1: Chunk starts before the start of Rec
        if chunk_start < 0:
            # Worst Case 1: The Buffer goes beyond(before) the start of entire recording
            if rec_num == 0:
                reduced_buffer = lpf_buffer + chunk_start  # Reduced Buffer for the Low Pass Filter Step

                if reduced_buffer < 0:
                    chunk_array = kwd_rec_raw_data[:chunk_end, index]
                    chunk_index = [0, int((rec_start + epoch_end) + chunk_buffer)]  # Correct the Index
                    worst_case = 1.1

                    if verbose:
                        print('Worst Case: 1.1')
                        print('No LPF|DS Buffer at the Start of the Chunk')
                        print(f"Starting Buffer of the Epoch(Chunk) is reduced by: {reduced_buffer/30000} seconds")
                else:
                    chunk_array = kwd_rec_raw_data[:chunk_end, index]
                    worst_case = 1

                    if verbose:
                        print('Worst Case: 1')
                        print('Not enough of a Buffer for the First Recording')
                        print(f"Starting Filter Buffer is reduced to: {reduced_buffer/30000} seconds")

            else:
                # Get the Prior Recording's Data
                prior_kwd_rec_raw_data = kwd_file['recordings'][str(rec_num - 1)]['data']
                # Stitch the starting samples with the prior Recording
                chunk_array[:np.abs(chunk_start), :] = prior_kwd_rec_raw_data[chunk_start:, index]
                # Stitch the ending samples with the current Recording
                chunk_array[np.abs(chunk_start):, :] = kwd_rec_raw_data[:chunk_end, index]

                if verbose:
                    print(f'Special Case 1: Recording {rec_num} to Recording {rec_num-1}')

        # Case 2: Chunk ends after the end of the Entire Days Recording
        elif chunk_end > kwd_rec_raw_data.shape[0]:
            relative_chunk_end = chunk_end - kwd_rec_raw_data.shape[0]  # Get Ending of Epoch in the Next Recording

            # Worst Case 2.1: The Buffer goes beyond the end of the entire recording
            if rec_num == int(max(kwd_file['recordings'].keys())):
                reduced_buffer = lpf_buffer - relative_chunk_end  # Reduced Buffer for the Low Pass Filter Step
                if reduced_buffer < 0:
                    chunk_array = kwd_rec_raw_data[chunk_start:, index]
                    chunk_index = \
                        [int((rec_start + epoch_start) - chunk_buffer), int(kwd_rec_raw_data.shape[0] + rec_start)]
                    worst_case = 2.1
                    if verbose:
                        print('Worst Case 2.1')
                        print('No LPF|DS Buffer at the End of the Chunk')
                        print(f"End Buffer of the Epoch(Chunk) is reduced by: {reduced_buffer/30000} seconds")

                # Worst Case 2: The LPF is Reduced
                else:
                    chunk_array = kwd_rec_raw_data[chunk_start:, index]
                    worst_case = 2
                    if verbose:
                        print('Worst Case 2')
                        print('Not enough of a Buffer for the Last Recording')
                        print(f"End Filter Buffer is reduced to: {reduced_buffer/30000} seconds")

            else:
                # Get the Next Recordings Data
                next_kwd_rec_raw_data = kwd_file['recordings'][str(rec_num + 1)]['data']
                # Stitch the starting samples with the prior Recording
                chunk_array[:-relative_chunk_end, :] = kwd_rec_raw_data[chunk_start:, index]
                # Stitch the ending samples with the current Recording
                chunk_array[-relative_chunk_end:, :] = next_kwd_rec_raw_data[:relative_chunk_end, index]
                if verbose:
                    print(f'Special Case 2: Recording {rec_num} to Recording {rec_num+1}')

        if len(index) == 1:  # If Singular Index
            chunk_array = chunk_array[:, 0] * .195  # Make the Correct Shape for mne with 0.195 µV resolution
        else:
            chunk_array = np.transpose(chunk_array * .195)  # Make the Correct Shape for mne with 0.195 µV resolution

    # print(epoch_start, 'to', epoch_end)  # Recording Number Motif Occurs During
    else:
        if len(index) == 1:  # If Singular Index
            chunk_array = kwd_rec_raw_data[chunk_start:chunk_end, index] * .195  # 0.195 µV resolution
            chunk_array = chunk_array[:, 0]  # Make the Correct Shape (1-darray)
        else:
            chunk_array = np.transpose(kwd_rec_raw_data[chunk_start:chunk_end, index]) * .195  # 0.195 µV resolution

    # Create the Absolute Index Entry that aren't edge cases with reduced Buffers
    if worst_case != 1.1 and worst_case != 2.1:
        chunk_index = [int((rec_start + epoch_start) - chunk_buffer), int((rec_start + epoch_end) + chunk_buffer)]
        if worst_case == 0:
            reduced_buffer = None

    return chunk_array, chunk_index, worst_case, reduced_buffer


def epoch_bpf_audio(kwd_file, kwe_data, chunks, audio_chan: list, filter_buffer: int = 10, data_buffer: int = 30,
                    verbose: bool = False):
    """Chunk the Audio and Bandpass Filter to remove noise

    Parameters
    ----------
    kwd_file : h5py.File
        KWD file imported using h5py library
    kwe_data : dict
        dictionary of the events in the KWE file
        Keys:
            'motif_st': [# of Motifs]
            'motif_rec_num': [# of Motifs]
    chunks : list
        array of the automated motif labels to use as benchmarks for the long epochs
    audio_chan: list
        list of the channel(s)[column(s)] of the .kwd file that are audio channels
    filter_buffer : int, optional
        Time buffer in secs to be sacrificed for filtering, defaults to 10 secs
    data_buffer : int, optional
        Time buffer around time of interest to chunk data, defaults to 30 secs
    verbose : bool
        If True the Function prints out useful statements, defaults to False

    Returns
    -------
    audio_chunks : list, shape = [Chunk]->(channels, Samples)
        Audio Data that is Bandpass Filtered between 300 and 10000 Hz, list of 2darrays

    Notes
    -----
        The raw data are saved as signed 16-bit integers, in the range -32768 to 32767. They don’t have a unit. To
    convert to microvolts, just  multiply by 0.195. This scales the data into the range ±6.390 mV,
    with 0.195 µV resolution (Intan chips have a ±5 mV input range).

    """

    fs = 30000  # Sampling Rate
    filt_buffer = filter_buffer * fs  # 10 sec Buffer for the Lowpass Filter
    chunk_buffer = data_buffer * fs  # 30 sec Buffer for Epoching

    audio_chunks = []

    for index, (start, end) in enumerate(chunks):
        if end is None:
            end = start

        if verbose:
            # Print out info about motif
            print('On Motif ', (index + 1), '/', len(chunks), 'Duration: ', )

        chunk_array, _, case_id, reduced_buffer = get_chunk_from_kwd(start=start, end=end, chunk_buffer=chunk_buffer,
                                                                     lpf_buffer=filt_buffer, kwd_file=kwd_file,
                                                                     kwe_data=kwe_data, index=audio_chan,
                                                                     verbose=verbose)

        # TODO: Rewrite Audio Filter Step with a Filter made for Audio
        chunk_filt = mne.filter.filter_data(chunk_array, sfreq=fs, l_freq=300, h_freq=10000, fir_design='firwin2',
                                            verbose=False)
        if len(audio_chan) == 1:
            # Remove The Extra filter Buffer
            if case_id == 0:
                audio_chunks.append(chunk_filt[filt_buffer:-filt_buffer])  # Base Case: It Fits in the entire Recording
            elif case_id == 1:
                audio_chunks.append(chunk_filt[reduced_buffer:-filt_buffer])  # Starting Filter buffer is clipped off
            elif case_id == 1.1:
                audio_chunks.append(chunk_filt[:-filt_buffer])  # Entire Starting Filter Buffer is gone
            elif case_id == 2:
                audio_chunks.append(chunk_filt[filt_buffer:-reduced_buffer])  # Ending filter buffer is clipped off
            elif case_id == 2.1:
                audio_chunks.append(chunk_filt[filt_buffer:])  # Entire ending filter buffer is gone

        else:  # Planning ahead for multiple Audio Channels
            # Remove The Extra filter Buffer
            if case_id == 0:
                audio_chunks.append(
                    chunk_filt[:, filt_buffer:-filt_buffer])  # Base Case: It Fits in the entire Recording
            elif case_id == 1:
                audio_chunks.append(chunk_filt[:, reduced_buffer:-filt_buffer])  # Starting Filter buffer is clipped off
            elif case_id == 1.1:
                audio_chunks.append(chunk_filt[:, -filt_buffer])  # Entire Starting Filter Buffer is gone
            elif case_id == 2:
                audio_chunks.append(chunk_filt[:, filt_buffer:-reduced_buffer])  # Ending filter buffer is clipped off
            elif case_id == 2.1:
                audio_chunks.append(chunk_filt[:, filt_buffer:])  # Entire ending filter buffer is gone
    return audio_chunks

# TODO: Make a Test for index to check there aren't any numbers less than 0 or greator than the Total len of recording


def epoch_lfp_ds_data(kwd_file, kwe_data, chunks, neural_chans: list, filter_buffer: int = 10, data_buffer: int = 30,
                      verbose: bool = False):
    """ Epochs Neural Data from the KWD File and converts it to µV, Low-Pass Filters and Downsamples to 1 KHz

        Parameters
        ----------
        kwd_file : h5py.File
            KWD file imported using h5py library
        kwe_data : dict
            dictionary of the events in the KWE file
            Keys:
                'motif_st': [# of Motifs]
                'motif_rec_num': [# of Motifs]
        chunks : list
            array of the automated motif labels to use as benchmarks for the long epochs
        neural_chans: list
            list of the channels(columns) of the .kwd file are neural channels
        filter_buffer : int, optional
            Time buffer in secs to be sacrificed for filtering, defaults to 10 secs
        data_buffer : int, optional
            Time buffer around time of interest to chunk data, defaults to 30 secs
        verbose : bool
            If True the Function prints out useful statements, defaults to False

        Returns
        -------
        neural_chunks : shape = [Chunk]->(channels, Samples)
            Neural Data that is Low-Pass Filter at 400 Hz and Downsampled to 1 KHz, list of 2darrays
        chunk_index : shape = [Chunk]->(absolute start, absolute end)
            List of the Absolute Start and End of Each Chunk for that Recordings Day

        Notes
        -----
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
    lpf_buffer = filter_buffer * fs  # 10 sec Buffer for the Lowpass Filter
    chunk_buffer = data_buffer * fs  # 30 sec Buffer for Epoching

    neural_chunks = []
    chunk_index = []

    for index, (start, end) in enumerate(chunks):
        if end is None:
            end = start

        if verbose:
            # Print out info about motif
            print('On Motif ', (index + 1), '/', len(chunks))

        chunk_array, index_single, case_id, reduced_buffer = get_chunk_from_kwd(start=start, end=end,
                                                                                chunk_buffer=chunk_buffer,
                                                                                lpf_buffer=lpf_buffer,
                                                                                kwd_file=kwd_file,
                                                                                kwe_data=kwe_data, index=neural_chans,
                                                                                verbose=verbose)

        chunk_index.append(index_single)  # Append the Absolute Index [Start, End] of the Chunk

        chunk_filt = mne.filter.filter_data(chunk_array, sfreq=fs, l_freq=None, h_freq=400, verbose=False)

        # Remove the LPF Buffer and Downsample to 1KHz
        if case_id == 0:
            neural_chunks.append(chunk_filt[:, lpf_buffer:-lpf_buffer:30])  # Base Case: It Fits in the entire Recording
        elif case_id == 1:
            neural_chunks.append(chunk_filt[:, reduced_buffer:-lpf_buffer:30])  # Starting Filter buffer is clipped off
        elif case_id == 1.1:
            neural_chunks.append(chunk_filt[:, :-lpf_buffer:30])  # the entire Starting Filter Buffer is gone
        elif case_id == 2:
            neural_chunks.append(chunk_filt[:, lpf_buffer:-reduced_buffer:30])  # Ending filter buffer is clipped off
        elif case_id == 2.1:
            neural_chunks.append(chunk_filt[:, lpf_buffer::30])  # the entire ending filter buffer is gone
    return neural_chunks, chunk_index



