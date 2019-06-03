
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
    """
    times_sorted = np.argsort(times)
    ledger = list(times_sorted[1:])
    logged = []
    focus = times_sorted[0]  # Select the second Candidate for the End of the Chunk
    candidate = times_sorted[1]  # Select the First Candidate for the End of the Chunk
    counter = 0

    while len(ledger) > 0:
        competitor = ledger[0]  # Select the Candidate for the End of the Chunk
        distance = (times[competitor] - times[focus]) / 30000  # Distance in Seconds

        if distance < 2:
            #         print(f"if: {distance}")
            candidate = ledger.pop(0)  # Update with the new candidate
            counter = counter + 1  # Count the Number of Motifs Contained in the Motif

        elif distance < 30:
            #         print(f"elif: {distance}")
            candidate = ledger.pop(0)  # Update with the new candidate
            counter = counter + 1  # Count the Number of Motifs Contained in the Motif

        else:
            #         print(f"else: {distance}")
            print(counter)
            if counter > 0:
                logged.append([focus, candidate])  # Save the Start and End of the Sucessful Chunk Pair
                focus = competitor  # Update the New Start Focus to the competitor
                ledger.pop(0)  # Clear the roster for the New Candidate
                candidate = ledger.pop(0)  # Update the New Start Candidated
                counter = 0
            else:
                logged.append([focus, None])  # Special Case Lone Motif
                focus = competitor  # Update the New Start Focus to the competitor
                ledger.pop(0)  # Clear the roster for the New Candidate
                candidate = ledger.pop(0)  # Update the New Start Candidated
                counter = 0

    return logged
