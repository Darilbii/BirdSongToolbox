""" Functions for Running Classification and Prediction analysis labeled Chunks of Free Behavior"""

import numpy as np
import scipy
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


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


def ml_selector(event_data, identity_index, label_index, sel_instances):
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


def pearson_extraction(event_data, templates):
    """  Pearson Correlation Coefficients for all Labels

    Parameters
    ----------
    event_data : ndarray | (labels, instances, frequencies, channels, samples)
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
    for label in event_data:
        extracted_pearson.append(find_pearson_coeff(label, templates=templates))
    return np.asarray(extracted_pearson)


def make_feature_id_ledger(num_freqs, num_chans, num_temps):
    """ Make a Feature Identity Ledger for One type of Feature rows=[Freqs, Channels] | (2, Num_Features)

    Parameters
    ----------
    num_freqs : int
        the number of frequency bands
    num_chans : int
        the number of recording channels
    num_temps : int, optional
        the number of pearson templates, only include if the feature type is pearson

    Returns
    -------
    entries_ledger : ndarray | (num_total_features, [frequencies, channels, templates])
        ledger of the feature identity for the scikit-learn data structure

    """

    if num_temps:
        entries_ledger = np.zeros((3, num_freqs, num_chans, num_temps))
        # Make Ledger for Frequency
        entries_ledger[0] = entries_ledger[0] + np.arange(num_freqs)[:, None, None]  # Index of Freq
        # Make Ledger for Channel
        entries_ledger[1] = entries_ledger[1] + np.arange(num_chans)[None, :, None]  # Index of Channel
        # Make Ledger for templates
        entries_ledger[2] = entries_ledger[2] + np.arange(num_temps)[None, None, :]  # Index of Templates
        entries_ledger = np.asarray(entries_ledger)  # Convert to ndarray
        entries_ledger = entries_ledger.reshape((3, -1))  # Convert (3, Num_Features)

    else:
        entries_ledger = np.zeros((2, num_freqs, num_chans))
        # Make Ledger for Frequency
        entries_ledger[0, :, :] = entries_ledger[0, :, :] + np.arange(num_freqs)[:, None]  # Index of Freq
        # Make Ledger for Channel
        entries_ledger[1, :, :] = entries_ledger[1, :, :] + np.arange(num_chans)[None, :]  # Index of Channel
        entries_ledger = np.asarray(entries_ledger)  # Convert to ndarray
        entries_ledger = entries_ledger.reshape((2, -1))  # Convert (2, Num_Features)

    return np.transpose(entries_ledger)


def ml_order(extracted_features_array):
    """
    Parameters
    ----------
    extracted_features_array : ndarray | (labels, instances, frequencies, channels, templates)
        Array of Pearson Correlation Values between each instance and the LFP Template of each Label

    Returns
    -------
    ordered_trials : ndarray | (n_samples, n_features)
        Data array that is structured to work with the SciKit-learn Package
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    ml_labels : ndarray | (n_training_samples, )
        1-d array of Labels of the ordered_trials instances )(n_samples)
    """
    ml_labels = []
    ordered_trials = []
    for index, label in enumerate(extracted_features_array):
        # Machine Learning Data
        num_instances = len(label)
        reshaped = np.reshape(label, (num_instances, -1))
        ordered_trials.extend(reshaped)

        # Machine Learning Labels
        label_dummy = np.zeros((num_instances, 1))
        label_dummy[:] = index
        ml_labels.extend(label_dummy)

    ordered_trials = np.array(ordered_trials)
    ml_labels = np.array(ml_labels)[:, 0]

    return ordered_trials, ml_labels


def make_feature_dict(ordered_index, drop_type: str):
    """Creates a Dictionary of the the indexes for each Channel's features in the ordered_index

    Parameters
    ----------
    ordered_index : ndarray | (num_total_features, [frequencies, channels, templates])
        ledger of the feature identity for the scikit-learn data structure
    drop_type : str
        Controls whether the dictionary indexes the channel number of the frequency band
    Returns
    -------
    feature_dict : dict | {feature: [list of Indexes]}
        dictionary to be used to remove all features for either a single channel or frequency band

    """
    options = ['channel', 'frequency']
    assert drop_type in options

    if drop_type == 'frequency':
        sel = 0
    elif drop_type == 'channel':
        sel = 1

    ordered_index_shape = np.max(ordered_index, axis=0) + 1
    sel_len = int(ordered_index_shape[sel])

    feature_dict = {}
    for i in range(sel_len):
        feature_dict[i] = [index for index, description in enumerate(ordered_index) if description[sel] == i]

    return feature_dict


def drop_features(features, keys, desig_drop_list):
    """Function for Selectively Removing Columns for Feature Dropping

    Parameters
    ----------
    features : ndarray | (n_samples, n_features)
        Data array that is structured to work with the SciKit-learn Package
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    keys : dict | {feature: [list of Indexes]}
        dictionary to be used to remove all features for either a single channel or frequency band
    desig_drop_list : list
        list of features to be dropped

    Returns
    -------
    remaining_features : ndarray | (n_samples, n_features_remaining)
        Data array that is structured to work with the SciKit-learn Package
    full_drop : list
        list of list of all Features (columns) to be dropped

    """
    # flatten_matrix = [val
    #                   for sublist in matrix
    #                   for val in sublist]

    full_drop = [val for i in desig_drop_list for val in keys[i]]  # Store the Index of Features to be dropped

    remaining_features = np.delete(features, full_drop, axis=1)

    return remaining_features, full_drop


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


def random_feature_dropping(train_set: np.ndarray, train_labels: np.ndarray, test_set: np.ndarray,
                            test_labels: np.ndarray, ordered_index, drop_type,  Class_Obj, verbose=False):
    """ Repeatedly trains/test models to create a feature dropping curve (Originally for Pearson Correlation)

    Parameters
    ----------
    train_set : ndarray | (n_samples, n_features)
        Training data array that is structured to work with the SciKit-learn Package
    train_labels : ndarray | (n_training_samples, )
        1-d array of Labels of the Corresponding n_training_samples
    test_set : ndarray  | (n_samples, n_features)
        Testing data Array that is structured to work with the SciKit-learn Package
    test_labels : ndarray | | (n_training_samples, )
        1-d array of Labels of the Corresponding n_test_samples
    ordered_index : ndarray | (num_total_features, [frequencies, channels, templates])
        ledger of the feature identity for the scikit-learn data structure
            Power:   (Num of Features, [frequencies, channels])
            Pearson: (Num of Features, [frequencies, channels, templates])
    drop_type : str
        Controls whether the dictionary indexes the channel number of the frequency band
    Class_Obj : class
        classifier object from the scikit-learn package
    verbose : bool
        If True the funtion will print out useful information for user as it runs, defaults to False.

    Returns
    -------
    dropping_curve : ndarray
        ndarray of accuracy values from the feature dropping code (values are floats)
        (Number of Features (Decreasing), Number of Nested Folds)
    """

    # 1.) Initiate Lists for Curve Components
    feat_ids = make_feature_dict(ordered_index=ordered_index, drop_type=drop_type)  # Convert ordered_index to a dict
    num_channels = len(feat_ids.keys())  # Determine the Number of Dropping indexes
    dropping_curve = np.zeros([num_channels + 1, 1])  # Create Empty array for Dropping Curves
    drop_list = []

    # 2.) Print Information about the Feature Set to be Dropped
    if verbose:
        print("Number of columns dropped per cycle", len(feat_ids[0]))  # Print number of columns per dropped feature
        print("Number of Channels total:", len(feat_ids))  # Print number of Features

    temp = feat_ids.copy()  # Create a temporary internal *shallow? copy of the index dictionary

    # 3.) Begin Feature Dropping steps
    # Find the first Accuracy

    first_acc, _, _ = clip_classification(ClassObj=Class_Obj, train_set=train_set, train_labels=train_labels,
                                          test_set=test_set, test_labels=test_labels)

    if verbose:
        print("First acc: %s..." % first_acc)
        # print("First Standard Error is: %s" % first_err_bars)  ###### I added this for the error bars

    dropping_curve[0, :] = first_acc  # Append BDF's Accuracy to Curve List
    index = 1

    while num_channels > 2:  # Decrease once done with development
        ids_remaining = list(temp.keys())  # Make List of the Keys(Features) from those that remain
        num_channels = len(ids_remaining)  # keep track of the number of Features
        # Select the index for Feature to be Dropped from list of keys those remaining (using random.choice())
        drop_feat_ids = random.choice(ids_remaining)

        if verbose:
            print("List of Channels Left: ", ids_remaining)
            print("Number of Channels Left:", num_channels)
            print("Channel to be Dropped:", drop_feat_ids)

        # Remove Key and Index for Designated Feature
        del temp[drop_feat_ids]  # Delete key for Feature Designated to be Dropped from overall list

        drop_list.append(drop_feat_ids)  # Add Designated Drop Feature to Drop list

        # Remove sel feature from train feature array
        train_remaining_features, _ = drop_features(features=train_set, keys=feat_ids, desig_drop_list=drop_list)

        # Remove sel feature from test feature array
        test_remaining_features, _ = drop_features(features=test_set, keys=feat_ids, desig_drop_list=drop_list)

        acc, _, _ = clip_classification(ClassObj=Class_Obj, train_set=train_remaining_features,
                                        train_labels=train_labels, test_set=test_remaining_features,
                                        test_labels=test_labels)

        dropping_curve[index, :] = acc  # Append Resulting Accuracy to Curve List

        if verbose:
            print("Drop accuracies: ", acc)
            print("Dropping Feature was %s..." % drop_feat_ids)

        index += 1

    return dropping_curve


def random_feature_drop_multi_narrow_chunk(event_data, ClassObj, k_folds=5, seed=None, verbose=False):
    """ Runs the Random Channel Feature Dropping algorithm on a set of pre-processed data

    Parameters
    ----------
    event_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)
    ClassObj : class
        classifier object from the scikit-learn package
    k_folds : int
        Number of Folds to Split between Template | Train/Test sets, defaults to 5,
    seed : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    verbose : bool
        If True the funtion will print out useful information for user as it runs, defaults to False.

    Returns
    -------
    """

    # 1.) Make Array for Holding all of the feature dropping curves
    nested_dropping_curves = []  # np.zeros([])

    # 2.) Create INDEX of all instances of interests : create_discrete_index()
    label_identities, label_index = create_discrete_index(event_data=event_data)
    identity_index = np.arange(len(label_index))
    sss = StratifiedShuffleSplit(n_splits=k_folds, random_state=seed)
    sss.get_n_splits(identity_index, label_index)

    if verbose:
        print(sss)

    # --------- For Loop over possible Training Sets---------
    for train_index, test_index in sss.split(identity_index, label_index):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = identity_index[train_index], identity_index[test_index]
        y_train, y_test = label_index[train_index], label_index[test_index]

        # 4.) Use INDEX to Break into corresponding [template/training set| test set] : ml_selector()
        # 4.1) Get template set/training : ml_selector(event_data, identity_index, label_index, sel_instances)
        sel_train = ml_selector(event_data=event_data, identity_index=label_identities, label_index=label_index,
                                sel_instances=X_train,)

        # 4.1) Get test set : ml_selector()
        sel_test = ml_selector(event_data=event_data, identity_index=label_identities, label_index=label_index,
                               sel_instances=X_test)

        # 5.) Use template/training set to make template : make_templates(event_data)
        templates = make_templates(event_data=sel_train)

        # 6.1) Use template/training INDEX and template to create Training Pearson Features : pearson_extraction()
        train_pearson_features = pearson_extraction(event_data=sel_train, templates=templates)

        # 6.2) Use test INDEX and template to create Test Pearson Features : pearson_extraction()
        test_pearson_features = pearson_extraction(event_data=sel_test, templates=templates)

        # 7.1) Reorganize Test Set into Machine Learning Format : ml_order_pearson()
        ml_trials_train, ml_labels_train = ml_order(extracted_features_array=train_pearson_features)

        # 7.2) Get Ledger of the Features
        num_freqs, num_chans, num_temps = np.shape(train_pearson_features[0][0])  # Get the shape of the Feature data
        ordered_index = make_feature_id_ledger(num_freqs=num_freqs, num_chans=num_chans, num_temps=num_temps)

        # 7.3) Reorganize Training Set into Machine Learning Format : ml_order_pearson()
        ml_trials_test, ml_labels_test = ml_order(extracted_features_array=test_pearson_features)

        fold_frequency_curves = []
        for freq in range(num_freqs):
            if verbose:
                print("On Frequency Band:", freq, " of:", num_freqs)

            ml_trials_train_cp = ml_trials_train.copy()  # make a copy of the feature extracted Train data
            ml_trials_test_cp = ml_trials_test.copy()  # make a copy of the feature extracted Test data
            ordered_index_cp = ordered_index.copy()  # make a copy of the ordered_index
            all_other_freqs = list(np.delete(np.arange(num_freqs), [freq])) # Make a index of the other frequencies
            temp_feature_dict = make_feature_dict(ordered_index=ordered_index_cp, drop_type='frequency')  # Feature Dict
            # reduce to selected frequency from the COPY of the training data
            ml_trials_train_freq, full_drop = drop_features(features=ml_trials_train_cp, keys=temp_feature_dict,
                                                 desig_drop_list=all_other_freqs)
            # reduce to but the selected frequency from the COPY of test data
            ml_trials_test_freq, _ = drop_features(features=ml_trials_test_cp, keys=temp_feature_dict,
                                                desig_drop_list=all_other_freqs)
            ordered_index_cp = np.delete(ordered_index_cp, full_drop, axis=0)  # Remove features from other frequencies

            # 8.) Perform Nested Feature Dropping with K-Fold Cross Validation
            nested_drop_curve = random_feature_dropping(train_set=ml_trials_train_freq, train_labels=ml_labels_train,
                                                        test_set=ml_trials_test_freq, test_labels=ml_labels_test,
                                                        ordered_index=ordered_index_cp, drop_type='channel',
                                                        Class_Obj=ClassObj, verbose=False)
            fold_frequency_curves.append(nested_drop_curve)

        nested_dropping_curves.append(fold_frequency_curves)

    # 9.) Combine all curve arrays to one array
    all_drop_curves = np.array(nested_dropping_curves)  # (folds, frequencies, num_dropped, 1)


    # 10.) Calculate curve metrics
    mean_curve = np.mean(all_drop_curves, axis=0)
    # std_curve = np.std(all_drop_curves, axis=0, ddof=1)  # ddof parameter is set to 1 to return the sample std
    std_curve = scipy.stats.sem(all_drop_curves, axis=0)

    return mean_curve, std_curve


def random_feature_drop_chunk(event_data, ClassObj, k_folds=5, seed=None, verbose=False):
    """ Runs the Random Channel Feature Dropping algorithm on a set of pre-processed data (All Features Together)

    Parameters
    ----------
    event_data : ndarray | (classes, instances, frequencies, channels, samples)
        Randomly Rebalanced Neural Data (output of balance_classes)
    ClassObj : class
        classifier object from the scikit-learn package
    k_folds : int
        Number of Folds to Split between Template | Train/Test sets, defaults to 5,
    seed : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    verbose : bool
        If True the function will print out useful information for user as it runs, defaults to False.

    Returns
    -------
    """

    # 1.) Make Array for Holding all of the feature dropping curves
    nested_dropping_curves = []  # np.zeros([])

    # 2.) Create INDEX of all instances of interests : create_discrete_index()
    label_identities, label_index = create_discrete_index(event_data=event_data)
    identity_index = np.arange(len(label_index))
    sss = StratifiedShuffleSplit(n_splits=k_folds, random_state=seed)
    sss.get_n_splits(identity_index, label_index)

    if verbose:
        print(sss)

    # --------- For Loop over possible Training Sets---------
    for train_index, test_index in sss.split(identity_index, label_index):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = identity_index[train_index], identity_index[test_index]
        y_train, y_test = label_index[train_index], label_index[test_index]

        # 4.) Use INDEX to Break into corresponding [template/training set| test set] : ml_selector()
        # 4.1) Get template set/training : ml_selector(event_data, identity_index, label_index, sel_instances)
        sel_train = ml_selector(event_data=event_data, identity_index=label_identities, label_index=label_index,
                                sel_instances=X_train,)

        # 4.1) Get test set : ml_selector()
        sel_test = ml_selector(event_data=event_data, identity_index=label_identities, label_index=label_index,
                               sel_instances=X_test)

        # 5.) Use template/training set to make template : make_templates(event_data)
        templates = make_templates(event_data=sel_train)

        # 6.1) Use template/training INDEX and template to create Training Pearson Features : pearson_extraction()
        train_pearson_features = pearson_extraction(event_data=sel_train, templates=templates)

        # 6.2) Use test INDEX and template to create Test Pearson Features : pearson_extraction()
        test_pearson_features = pearson_extraction(event_data=sel_test, templates=templates)

        # 7.1) Reorganize Test Set into Machine Learning Format : ml_order_pearson()
        ml_trials_train, ml_labels_train = ml_order(extracted_features_array=train_pearson_features)

        # 7.2) Get Ledger of the Features
        num_freqs, num_chans, num_temps = np.shape(train_pearson_features[0][0])  # Get the shape of the Feature data
        ordered_index = make_feature_id_ledger(num_freqs=num_freqs, num_chans=num_chans, num_temps=num_temps)

        # 7.3) Reorganize Training Set into Machine Learning Format : ml_order_pearson()
        ml_trials_test, ml_labels_test = ml_order(extracted_features_array=test_pearson_features)

        # 8.) Perform Nested Feature Dropping with K-Fold Cross Validation
        nested_drop_curve = random_feature_dropping(train_set=ml_trials_train, train_labels=ml_labels_train,
                                                    test_set=ml_trials_test, test_labels=ml_labels_test,
                                                    ordered_index=ordered_index, drop_type='channel',
                                                    Class_Obj=ClassObj, verbose=False)

        nested_dropping_curves.append(nested_drop_curve)

    # 9.) Combine all curve arrays to one array
    all_drop_curves = np.array(nested_dropping_curves)  # (folds, frequencies, num_dropped, 1)


    # 10.) Calculate curve metrics
    mean_curve = np.mean(all_drop_curves, axis=0)
    # std_curve = np.std(all_drop_curves, axis=0, ddof=1)  # ddof parameter is set to 1 to return the sample std
    std_curve = scipy.stats.sem(all_drop_curves, axis=0)

    return mean_curve, std_curve
