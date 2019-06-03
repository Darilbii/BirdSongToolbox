
import numpy as np


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


