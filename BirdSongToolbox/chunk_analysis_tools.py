""" Functions for Running Classification and Prediction analysis labeled Chunks of Free Behavior"""

import numpy as np
import scipy
from sklearn.metrics import confusion_matrix


def make_templates(event_data):
    """
    Parameters
    ----------
    event_data : ndarray | (classes, instances, frequencies, channels, samples)
        Neural Data that has been clipped around events of interest

    Returns
    -------
    templates : ndarray | (classes, frequencies, channels, samples)
        Mean of all instances for each label's Channel/Frequency pair
    """
    templates = []
    for data in event_data:
        label_template = np.mean(data, axis=0)  # label_data: (instances, frequencies, channels, samples)
        templates.append(label_template)
    return np.array(templates)


def ml_selector(event_data, identity_index, label_index, sel_instances, make_template=False, verbose=False):
    """ Collects Instances of Interest from the event_data

    Parameters
    ----------
    event_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)
    identity_index : ndarray | (num_instances_total,)
        array of indexes that represent the individual index of a class labels
    label_index : ndarray | (num_instances_total,)
        array of labels that indicates the class that instance is an example of
    sel_instances : ndarray | (number_instances_total,)
        array of indexes that represent the individual indexes of all instances across class labels

    make_template : bool
        If True function will return the mean of all instances for each label's Channel/Frequency pair, defaults to False
    verbose: bool
        If True the function will print out useful info about its progress, defaults to False

    Returns
    -------
    sel_data : ndarray | (classes, instances, frequencies, channels, samples)
         ndarray containing the Segments (aka clippings) designated by the sel_index parameter.
         Note: the number of instances aren't necessarily equal even if they are balanced prior to running this function
    """
    sel_id_index = identity_index[sel_instances]
    sel_label_index = label_index[sel_instances]

    sel_data = []
    for index, data in enumerate(event_data):
        label_instances = [x for x, y in zip(sel_id_index, sel_label_index) if y == index]  # Sel Instances for Label
        label_data = data[np.array(label_instances)]  # Array Index using the Selected Instances

        # if make_template:
        #     label_data = np.mean(label_data, axis=0)  # label_data: (instances, frequencies, channels, samples)

        sel_data.append(label_data)

    return np.array(sel_data)

def create_discrete_index(event_data):
    """
    Parameters
    ----------
    event_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)

    Returns
    -------
    identity_index : array | (num_instances_total,)
        array of indexes that represent the individual index of a class labels
    labels_index : array | (num_instances_total,)
        array of labels that indicates the class that instance is an example of
    """
    identity_index = []
    labels_index = []
    for index, sel_class in enumerate(event_data):
        label_dummy = np.zeros((sel_class.shape[0], 1))
        label_dummy[:] = index
        instance_dummy = np.arange(sel_class.shape[0])
        identity_index.extend(instance_dummy)
        labels_index.extend(label_dummy)
    identity_index = np.asarray(identity_index)  # Convert to ndarray
    labels_index = np.asarray(labels_index)[:, 0]  # Convert to a ndarray

    return identity_index, labels_index


def efficient_pearson_1d_v_2d(one_dim, two_dim):
    """Finds the Pearson correlation of all rows of the two dimensional array with the one dimensional array

    Source:
    -------
        https://www.quora.com/How-do-I-calculate-the-correlation-of-every-row-in-a-2D-array-to-a-1D-array-of-the-same-length

    Info:
    -----
        The Pearson correlation coefficient measures the linear relationship
     between two datasets. Strictly speaking, Pearson's correlation requires
     that each dataset be normally distributed. Like other correlation
     coefficients, this one varies between -1 and +1 with 0 implying no
     correlation. Correlations of -1 or +1 imply an exact linear
     relationship. Positive correlations imply that as x increases, so does
     y. Negative correlations imply that as x increases, y decreases.


    Parameters
    ----------
    one_dim = ndarray | (samples,)
        1-Dimensional Array
    two_dim= ndarray | (instances, samples)
        2-Dimensional array it's row length must be equal to the length of one_dim

    Returns
    -------
    pearson_values : ndarray | (samples,)
        Pearson Correlation Values for each instance

    Example
    -------
    x = np.random.randn(10)
    y = np.random.randn(100, 10)

    The numerators is shape (100,) and denominators is shape (100,)
    Pearson = efficient_pearson_1d_v_2d(one_dim = x, two_dim = y)
    """
    x_bar = np.mean(one_dim)
    x_intermediate = one_dim - x_bar
    y_bar = np.mean(two_dim, axis=1)  # this flattens y to be (100,) which is a 1D array.
    # The problem is that y is 100, so numpy's broadcasting doesn't know which axis to choose to broadcast over.
    y_bar = y_bar[:, np.newaxis]
    # By adding this extra dimension, we're forcing numpy to treat the 0th axis as the one to broadcast over
    # which makes the next step possible. y_bar is now 100, 1
    y_intermediate = two_dim - y_bar
    numerators = y_intermediate.dot(x_intermediate)  # or x_intermediate.dot(y_intermediate.T)
    x_sq = np.sum(np.square(x_intermediate))
    y_sqs = np.sum(np.square(y_intermediate), axis=1)
    denominators = np.sqrt(x_sq * y_sqs)  # scalar times vector
    pearson_values = (numerators / denominators)  # numerators is shape (100,) and denominators is shape (100,)

    return pearson_values


def find_pearson_coeff(cl_data, templates, slow=False):
    """ Iterates over each Template and finds the Pearson Coefficient for 1 template at a time

        Note: This Function Mirrors find_power() only for finding Pearson Correlation Coefficient

    Information
    -----------
    Note : The Number of Examples of Label does not always equal the total number of examples total as some push past
        the time frame of the Epoch and are excluded

    Parameters
    ----------
    cl_data : ndarray | (instances, frequencies, channels, samples)
        Array containing all the neural segments for one labels.
        (As defined by Label_Instructions in label_extract_pipeline)
    templates : ndarray | (labels, frequencies, channels, samples)
        Array of Template Neural Data that corresponds to the Label designated (Templates are the mean of trials)
    slow : bool, optional
        if True the code will use the native scipy.stats.pearsonr() function which is slow (defaults to False)

    Returns
    -------
    corr_trials : ndarray | (instances, frequencies, channels, labels/templates)
        Array of Pearson Correlation Values between each instance and the LFP Template of each Label
    """

    num_instances, num_frequencies, num_channels, trial_length = np.shape(cl_data)
    num_temps = len(templates)

    # Create Lists
    corr_trials = []

    if slow:
        for frequency in range(num_frequencies):  # Over all Frequency Bands
            channel_trials = []
            for channel in range(num_channels):  # For each Channel
                corr_holder = np.zeros([num_instances, num_temps])

                for instance in range(num_instances):
                    for temp in range(num_temps):
                        corr_holder[instance, temp], _ = scipy.stats.pearsonr(cl_data[instance, frequency, channel, :],
                                                                              templates[temp, frequency, channel, :])
                channel_trials.append(corr_holder)  # Save all of the Trials for that Frequency on that Channel
            corr_trials.append(channel_trials)  # Save all of the Trials for all Frequencies on each Channel

    else:
        for frequency in range(num_frequencies):  # Over all Frequency Bands
            channel_trials = []
            for channel in range(num_channels):  # For each Channel
                corr_holder = np.zeros([num_instances, num_temps])
                for temp in range(num_temps):
                    corr_holder[:, temp] = efficient_pearson_1d_v_2d(templates[temp, frequency, channel, :],
                                                                     cl_data[:, frequency, channel, :])
                channel_trials.append(corr_holder)  # Save all of the Trials for that Frequency on that Channel
            corr_trials.append(channel_trials)  # Save all of the Trials for all Frequencies on each Channel

    corr_trials = np.array(corr_trials)
    corr_trials = np.transpose(corr_trials, [2, 0, 1, 3])
    return corr_trials


def pearson_extraction(clipped_data, templates):
    """  Pearson Correlation Coefficients for all Labels

    Parameters
    ----------
    clipped_data : ndarray | (labels, instances, frequencies, channels, samples)
        Array containing all the neural segments for all labels.
        (As defined by Label_Instructions in label_extract_pipeline)
    templates : ndarray | (labels, frequencies, channels, samples)
        Array of Template Neural Data that corresponds to the Label designated (Templates are the mean of trials)

    Returns
    -------
    extracted_pearson : ndarray | (labels, instances,  frequencies, channels, templates)
        Array of Pearson Correlation Values between each instance and the LFP Template of each Label
    """
    extracted_pearson = []
    for label in clipped_data:
        extracted_pearson.append(find_pearson_coeff(label, templates=templates))
    return np.asarray(extracted_pearson)


def make_channel_identity_ledger(number_frequencies, number_channels, number_templates):
    """ Make a Feature Identity Ledger for One type of Feature rows=[Freqs, Channels] | (2, Num_Features)"""

    if number_templates:
        entries_ledger = np.zeros((3, number_frequencies, number_channels, number_templates))
        # Make Ledger for Frequency
        entries_ledger[0] = entries_ledger[0] + np.arange(number_frequencies)[:, None, None]  # Index of Freq
        # Make Ledger for Channel
        entries_ledger[1] = entries_ledger[1] + np.arange(number_channels)[None, :, None]  # Index of Channel
        # Make Ledger for templates
        entries_ledger[2] = entries_ledger[2] + np.arange(number_templates)[None, None, :]  # Index of Templates
        entries_ledger = np.asarray(entries_ledger)  # Convert to ndarray
        entries_ledger = entries_ledger.reshape((3, -1))  # Convert (3, Num_Features)

    else:
        entries_ledger = np.zeros((2, number_frequencies, number_channels))
        # Make Ledger for Frequency
        entries_ledger[0, :, :] = entries_ledger[0, :, :] + np.arange(number_frequencies)[:, None]  # Index of Freq
        # Make Ledger for Channel
        entries_ledger[1, :, :] = entries_ledger[1, :, :] + np.arange(number_channels)[None, :]  # Index of Channel
        entries_ledger = np.asarray(entries_ledger)  # Convert to ndarray
        entries_ledger = entries_ledger.reshape((2, -1))  # Convert (2, Num_Features)

    return entries_ledger


# TODO: Test This
def ml_order(extracted_pearson):
    """
    Parameters
    ----------
    extracted_pearson : ndarray | (labels, instances, frequencies, channels, templates)
        Array of Pearson Correlation Values between each instance and the LFP Template of each Label

    """
    ml_labels = []
    ordered_trials = []
    for index, label in enumerate(extracted_pearson):
        # Machine Learning Data
        num_instances = len(label)
        reshaped = np.reshape(label, (num_instances, -1))
        ordered_trials.extend(reshaped)

        # Machine Learning Labels
        label_dummy = np.zeros((num_instances, 1))
        label_dummy[:] = index
        ml_labels.extend(label_dummy)

    ordered_trials = np.array(ordered_trials)
    ml_labels = np.array(ml_labels)

    return ordered_trials, ml_labels


def clip_classification(ClassObj, train_set, train_labels, test_set, test_labels):
    """ This Function is a Flexible Machine Learning Function that Trains One Classifier and determines metrics for it
    The metrics it determines are:
                [1] Accuracy
                [2] Confusion Matrix

    Parameters
    ----------
    ClassObj : class
        classifier object from the scikit-learn package
    train_set : ndarray | (n_samples, n_features)
        Training data array that is structured to work with the SciKit-learn Package
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    train_labels : ndarray | (n_training_samples, 1)
        1-d array of Labels of the Corresponding n_training_samples
    test_set : ndarray | (n_samples, n_features)
        Testing data Array that is structured to work with the SciKit-learn Package
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    test_labels : ndarray | (n_test_samples, 1)
        1-d array of Labels of the Corresponding n_test_samples


    Returns
    -------
    acc : int
        the accuracy of the trained classifier
    classifier : class
        a trained classifier dictacted by the ClassObj Parameter from scikit-learn
    confusion : array
        Confusion matrix, shape = [n_classes, n_classes]

    """

    classifier = ClassObj
    classifier.fit(train_set, train_labels)  # Train the Classifier
    test_pred = classifier.predict(test_set)  # Test the Classifier
    confusion = confusion_matrix(test_labels, test_pred).astype(float)  # Determine the Confusion mattrix
    num_test_trials = len(test_labels)  # Get the number of trials
    acc = sum(np.diag(confusion)) / num_test_trials # accuracy = number_-right/ total_number

    return acc, classifier, confusion


