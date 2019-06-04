
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


def get_chunk_lfp_data(kwd_file, kwe_data, song_len_ms, before_t):
    """ Gets Neural Information from the KWD File and converts it to µV

    Parameters:
    -----------
    kwd_file: h5py.File
        KWD file imported using h5py library
    kwe_data: dict
        dictionary of the events in the KWE file
        Keys:
            'motif_st': [# of Motifs]
            'motif_rec_num': [# of Motifs]
    song_len_ms: int
        Length of Time desired to be Grabbed for the Motif in ms
    before_t: int
        The amount of time (ms) before the  motif to start the data collection

    Returns:
    --------
    lfp: ndarray
        Multidimensional array of Neural Raw signal Recording
        (Motif Length in Samples  x  Num. of Channels  x  Num. of Motifs)

    Notes:
    ------
        The raw data are saved as signed 16-bit integers, in the range -32768 to 32767. They don’t have a unit. To
    convert to microvolts, just  multiply by 0.195. This scales the data into the range ±6.390 mV,
    with 0.195 µV resolution (Intan chips have a ±5 mV input range).

    """

    for chunks in range(kwe_data['motif_st'].shape[0]):

        # Get start time for motif and recording start
        motif_start_time = kwe_data['motif_st'][chunks]  # Start Time of Motif in its Specific Recording
        motif_rec_num = kwe_data['motif_rec_num'][chunks]  # Recording Number Motif Occurs During
        # motif_rec_start = kwik_data['recordingStarts'][kwe_data['motif_rec_num'][Motif]]  # Start Sample of Recording
        kwd_rec_raw_data = kwd_file['recordings'][str(motif_rec_num)]['data']  # Raw Data for this Recording Number

        # Get Start Time and End Time in samples for the motif
        start_time_lfp = int(motif_start_time - before_t * 30)
        end_time_lfp = int(start_time_lfp + song_len_ms * 30)

        # Print out info about motif
        print('On Motif ', (chunks + 1), '/', kwe_data['motif_st'].shape[0])

        num_kwd_ch = kwd_rec_raw_data.shape[1]

        if chunks == 0:
            lfp = np.zeros((kwd_rec_raw_data[start_time_lfp:end_time_lfp, 0:num_kwd_ch - 1].shape[0],
                            kwd_rec_raw_data[start_time_lfp:end_time_lfp, 0:num_kwd_ch - 1].shape[1],
                            kwe_data['motif_st'].shape[0]))
            lfp[:, :, chunks] = kwd_rec_raw_data[start_time_lfp:end_time_lfp, 0:num_kwd_ch - 1]
        else:
            lfp[:, :, chunks] = kwd_rec_raw_data[start_time_lfp:end_time_lfp, 0:num_kwd_ch - 1]

    lfp = lfp * .195  # Convert to µVs
    return lfp


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

        epoch_start = kwe_data['motif_st'][start]
        epoch_end = kwe_data['motif_st'][end]
        rec_num = kwe_data['motif_rec_num'][start]
        kwd_rec_raw_data = kwd_file['recordings'][str(rec_num)]['data']
        rec_start = kwik_data['recordingStarts'][kwe_data['motif_rec_num'][start]]  # Start Sample of Recording

        if verbose:
            # Print out info about motif
            print('On Motif ', (index + 1), '/', len(chunks))

        chunk_start = int(epoch_start - (chunk_buffer + lpf_buffer))
        chunk_end = int(epoch_end + chunk_buffer + lpf_buffer)
        chunk_index.append([int((rec_start + epoch_start) - chunk_buffer), int((rec_start + epoch_end) + chunk_buffer)])

        print(epoch_start, 'to', epoch_end)  # Recording Number Motif Occurs During
        chunk_array = np.transpose(kwd_rec_raw_data[chunk_start:chunk_end, :-1]) * .195
        chunk_filt = mne.filter.filter_data(chunk_array,
                                            sfreq=fs,
                                            l_freq=None,
                                            h_freq=400,
                                            verbose=False)
        buff_chunks.append(chunk_filt[:, lpf_buffer:-lpf_buffer:30])  # Remove the LPF Buffer and Downsample to 1KHz
    return buff_chunks, chunk_index


# def get_chunk_absolute_times():

######################################################################################################################
# ####################################################################################################################
# ##################################################################################################################


