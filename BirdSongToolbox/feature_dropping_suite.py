import random

from sklearn.model_selection import StratifiedShuffleSplit

# Functions created to handle a more discrete

import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.model_selection import StratifiedShuffleSplit
from BirdSongToolbox.Epoch_Analysis_Tools import clip_classification, make_channel_dict, Drop_Features, \
    label_extract_pipeline, pearson_extraction, ml_order_pearson, kfold_wrapper_rand


# Changes for new Feature Dropping Prep
# 1.) Break into all instances of interest : Label_Extract_Pipeline()
# 2.) Create INDEX of all instances of interests : create_discrete_index()
# --------- For Loop ---------
# 3.) Break INDEX into [(10%) template set| (90%) training/test set]
# 4.) Use INDEX to Break into corresponding [template set| training/test set] : ml_selector()
# 5.) Use template set to make template : ml_selector(make_template=True)
# 6.) Use training/test INDEX and template to create Pearson Features : pearson_extraction()
# 7.) Reorganize into Machine Learning Format : ml_order_pearson()
# --------- Nested Cross-Validation ---------
# 8.) Cross-Validated Training/Testing
# 9.) ?? Not sure if there is something after this

# 2.) Create INDEX of all instances of interests : create_discrete_index()

# Development of a more Discrete handling of instances so that training sets are equal


# CHANGE:

## Create index of the total number of of each class


def create_discrete_index(clippings, verbose=True):
    """ Creates an array of the class labels of all class instances in a day's recordings

    Parameters
    ----------
    clippings: list
        List containing the Segments Designated by the Label_Instructions, Offset, and Tr_Length parameters
        [labels] -> [ch] -> [freq] -> ( Samples x Label Instances)
    verbose: bool
        If True the function prints out useful info about its progress, defaults to True

    Returns
    -------
    discrete_identity_index: ndarray
        array of indexs within each labels (0 to number of instances for labels), 1-dimensional
            (Number of Instances Total,)
    discrete_index: ndarray
        array of class labels, 1-dimensional
            (Number of Instances Total,)
    """

    discrete_identity_index = np.array([0, 0])
    discrete_index = np.array([0, 0])

    for index, labels in enumerate(clippings):
        # Make Index of Instance Identities
        identities = np.arange(len(labels[0][0][0, :]))
        discrete_identity_index = np.concatenate((discrete_identity_index, identities), axis=0)

        # Make Index of Labels
        additions = np.ones(len(labels[0][0][0, :]))
        discrete_index = np.concatenate((discrete_index, additions * index), axis=0)

        if verbose:
            print(f"Class {index} has ", len(labels[0][0][0, :]), "exemplars")

    discrete_identity_index: ndarray = np.delete(discrete_identity_index, [0, 1], axis=0)
    discrete_index: ndarray = np.delete(discrete_index, [0, 1], axis=0)

    return discrete_identity_index, discrete_index


# 3.) Use INDEX to Break into corresponding [template set| training/test set] : NOT(convenient_selector())


def ml_selector(clippings, identity_index, label_index, sel_instances, make_template=False, verbose=False):
    """ Collects Instances of Interest from the full Clippings object

    Parameters:
    -----------
    clippings: list
        List containing the Segments Designated by the label_instructions, offset, and tr_Length parameters
        [labels] -> [ch] -> [freq] -> ( Samples x Label Instances)
    identity_index: ndarray
        array of indexes that represent the individual index of a class labels, 1-dimensional
        (Number of Instances Total,)
    label_index: ndarray
        array of labels that indicates the class that instance is an example of, 1-dimensional
        (Number of Instances Total,)
    sel_instances: ndarray
        array of indexes that represent the individual indexes of all instances across class labels, 1-dimensional
        (Number of Instances Total,)
    make_template: bool
        If True function will return the mean of all instances for each label's Channel/Frequency pair, defaults to False
    verbose: bool
        If True the function will print out useful info about its progress, defaults to False

    Returns:
    --------
    sel_trials: list
        If make_template is false:
            List containing the Segments (aka clippings) designated by the sel_index parameter
            [labels] -> [ch] -> [freq] -> (Samples x Label Instances)
        If make_template is True
            List containing the mean of all Segments (aka clippings) designated by the sel_index parameter
            [labels] -> [ch] -> [freq] -> (Samples x 1)
    """

    sel_trials = []

    for label_focus in range(len(clippings)):
        num_ch, num_freq, trial_length, _ = np.shape(clippings[label_focus])  # Get the structure of clippings
        chan_temp = []  # Create empty list for each channel

        if verbose:
            print("Now grabbing instances from class ", label_focus, "\n Number of Channels: ", num_ch,
                  "\n Number of Frequency Bands: ", num_freq, "\n Trial Length: ", trial_length)

        for chan in range(num_ch):
            freq_temp = []  # Create empty list for each frequency bin

            for freq in range(num_freq):
                # create empty array
                trials_holder = np.zeros((trial_length, len([x for x in label_index if x == label_focus])))
                instance_counter = 0
                if verbose:
                    tracker = []

                for identity, label in zip(sel_instances, label_index):
                    # Verify that the current label index is the label class we want
                    if label_focus == label:
                        trials_holder[:, instance_counter] = clippings[label_focus][chan][freq][:, identity_index[identity]]
                        instance_counter += 1
                        if verbose:
                            tracker.append(1)
                    else:
                        if verbose:
                            tracker.append(0)
                if make_template:
                    trials_holder = np.mean(trials_holder, axis=1)  # Find Means (Match Filter)
                    trials_holder = trials_holder.reshape(trial_length, 1)
                freq_temp.append(trials_holder)

            chan_temp.append(freq_temp)

        sel_trials.append(chan_temp)
        if verbose:
            print("Label Index Tracker: \n", tracker)

    return sel_trials


# def random_feature_dropping(dataset, labels, ordered_index, Class_Obj, k_folds=2, verbose=False, fold_verbose=False):
#     """ Repeatedly trains/test models to create a feature dropping curve (Originally for Pearson Correlation)
#
#     Parameters:
#     -----------
#     Data_Set: ndarray
#         Array that is structured to work with the SciKit-learn Package
#         (n_samples, n_features)
#             n_samples = Num of Instances Total
#             n_features = Num_Ch * Num_Freq)
#     Data_Labels: ndarray
#         1-d array of Labels of the Corresponding n_samples
#         ( n_samples   x   1 )
#     ordered_index: list
#         Index of Features for Feature Dropping
#                             [list] -> (Tuple)
#         Power:   [Num of Features] -> (Chan Num , Freq Num)
#         Pearson: [Num of Features] -> (Chan Num , Freq Num, Temp Num)
#     Class_Obj: class
#         classifier object from the scikit-learn package
#     k_folds: int (optional)
#         Number of Cross-validation folds to use, defaults to 2
#     verbose: bool
#         If True the funtion will print out useful information for user as it runs, defaults to False.
#     fold_verbose: bool
#         If True the Function will print out information about every Cross-Validation fold, defaults to False.
#
#     Returns:
#     --------
#     droppingCurve: ndarray
#         ndarray of accuracy values from the feature dropping code (values are floats)
#
#     """
#
#     # 1.) Initiate Lists for Curve Components
#     num_channels = ordered_index[-1][0] + 1  # Determine the Number of Channels (Assumes the ordered_index is in order)
#     dropping_curve = np.zeros([num_channels, 1])
#     features_list = list(np.arange(num_channels))
#     drop_list = []
#
#     feat_ids = make_channel_dict(ordered_index=ordered_index)  # Convert ordered_index to a dict to index feature drops
#
#     # 2.) Print Information about the Feature Set to be Dropped
#     print("Number of columns dropped per cycle", len(feat_ids[0]))  # Print number of columns per dropped feature
#     print("Number of Channels total:", len(feat_ids))  # Print number of Features
#
#     temp = feat_ids.copy()  # Create a temporary internal *shallow? copy of the index dictionary
#
#     # 3.) Begin Feature Dropping steps
#     # Find the first k-Fold Acc.
#     first_acc, _, _, _ = kfold_wrapper(Data_Set=dataset, Data_Labels=labels, k_folds=k_folds, Class_Obj=Class_Obj,
#                                        verbose=fold_verbose)
#
#     if verbose:
#         print("First acc: %s..." % first_acc)
#         # print("First Standard Error is: %s" % first_err_bars)  ###### I added this for the error bars
#
#     dropping_curve[0] = first_acc  # Append BDF's Accuracy to Curve List
#     index = 1
#
#     # 3.) Iterate over the Number of Channels randomly removing 1 until there is only one left
#     while num_channels > 1:  # Decrease once done with development
#         IDs = list(temp.keys())  # Make List of the Keys(Features)
#         print("List of Channels Left: ", IDs)
#
#         num_channels = len(IDs)  # keep track of the number of Features
#         print("Number of Channels Left:", num_channels)
#
#         # Determine Feature to be dropped (using Random.choice())
#         drop_feat_id = random.choice(features_list)  # Select the Designated Feature to be Dropped
#         del features_list[drop_feat_id]  # Remove the Designated Drop Feature from feature array
#         drop_list.append(drop_feat_id)  # Add Designated Drop Feature to Drop list
#         remaining_features, _ = drop_features(dataset, feat_ids, drop_list)  # Remove sel feature from feature array
#         acc, _, _, _ = kfold_wrapper(Data_Set=remaining_features, Data_Labels=labels, k_folds=k_folds,
#                                      Class_Obj=Class_Obj, verbose=fold_verbose)  # Record Prediction Accuracy
#
#         dropping_curve[index] = acc  # Append Resulting Accuracy to Curve List
#
#         if verbose:
#             print("Drop acc: %s..." % (acc))
#             print("Dropping Feature %s..." % drop_feat_id)
#
#         del temp[drop_feat_id]  # Delete key for BDF from Temp Dict
#         index += 1
#
#     return dropping_curve


## This needs to be a modular code that will conduct the feature dropping for one feature set
## Return (Number of Features (Decreasing or Increasing ?), Number of Nested Folds)

def random_feature_dropping(train_set: np.ndarray, train_labels: np.ndarray, test_set: np.ndarray, test_labels: np.ndarray, ordered_index, Class_Obj, verbose=False):
    """ Repeatedly trains/test models to create a feature dropping curve (Originally for Pearson Correlation)

    Parameters:
    -----------
    train_set: ndarray
        Training data array that is structured to work with the SciKit-learn Package
        (n_samples, n_features)
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    train_labels: ndarray
        1-d array of Labels of the Corresponding n_training_samples
        ( n_training_samples   x   1 )
    test_set: ndarray
        Testing data Array that is structured to work with the SciKit-learn Package
        (n_samples, n_features)
            n_samples = Num of Instances Total
            n_features = Num_Ch * Num_Freq)
    test_labels: ndarray
        1-d array of Labels of the Corresponding n_test_samples
        ( n_test_samples   x   1 )
    ordered_index: list
        Index of Features for Feature Dropping
                            [list] -> (Tuple)
        Power:   [Num of Features] -> (Chan Num , Freq Num)
        Pearson: [Num of Features] -> (Chan Num , Freq Num, Temp Num)
    Class_Obj: class
        classifier object from the scikit-learn package
    verbose: bool
        If True the funtion will print out useful information for user as it runs, defaults to False.

    Returns:
    --------
    dropping_curve: ndarray
        ndarray of accuracy values from the feature dropping code (values are floats)
        (Number of Features (Decreasing), Number of Nested Folds)

    """

    # 1.) Initiate Lists for Curve Components
    num_channels = ordered_index[-1][0] + 1  # Determine the Number of Channels (Assumes the ordered_index is in order)
    dropping_curve = np.zeros([num_channels + 1, 1])  # Create Empty array for Dropping Curves
    drop_list = []

    feat_ids = make_channel_dict(ordered_index=ordered_index)  # Convert ordered_index to a dict to index feature drops

    # 2.) Print Information about the Feature Set to be Dropped
    if verbose:
        print("Number of columns dropped per cycle", len(feat_ids[0]))  # Print number of columns per dropped feature
        print("Number of Channels total:", len(feat_ids))  # Print number of Features

    temp = feat_ids.copy()  # Create a temporary internal *shallow? copy of the index dictionary

    # 3.) Begin Feature Dropping steps
    # Find the first Accuracy

    first_acc, _, _ = clip_classification(Class_Obj=Class_Obj, Train_Set=train_set, Train_Labels=train_labels,
                                    Test_Set=test_set, Test_Labels=test_labels, verbose=verbose)

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
        train_remaining_features, _ = Drop_Features(train_set, feat_ids, drop_list)  # Remove sel feature from train feature array
        test_remaining_features, _ = Drop_Features(test_set, feat_ids, drop_list)  # Remove sel feature from test feature array

        acc, _, _ = clip_classification(Class_Obj=Class_Obj, Train_Set=train_remaining_features, Train_Labels=train_labels,
                                  Test_Set=test_remaining_features, Test_Labels=test_labels, verbose=verbose)

        dropping_curve[index, :] = acc  # Append Resulting Accuracy to Curve List

        if verbose:
            print("Drop accuracies: ", acc)
            print("Dropping Feature was %s..." % drop_feat_ids)

        index += 1

    return dropping_curve


#
# from scipy.stats import binom
#
#
# def get_chance(num_samples, num_classes, alpha=0.05, bon_correct=1):
#     """ Calculate statistically significant classifier performance for data set with limited samples
#
#     Parameters:
#     ----------
#     num_samples: array-like,
#         number of samples in data (assumed to be balanced)
#     num_classes: int
#         number of classes
#     alpha: float, optional
#         significance level given by z/n or the ratio of tolerated false positives, defaults to 0.05
#         (z: the number of observations correctly classified by chance, n: the number of all observations)
#     bon_correct: int, optional
#         The Bonferroni Correction. Set equal to the number of 'Test' being
#         for more information visit: https://en.wikipedia.org/wiki/Bonferroni_correction
#
#     Returns:
#     --------
#     base: float
#         Threshold for statistically significant classification rate, range [0, 1]
#
#     Example:
#     --------
#     > baseline_binom = getChance(num_samples=n_trials,num_classes=n_classes)
#
#     """
#
#     base = np.divide(binom.ppf(1 - (alpha / bon_correct), num_samples, 1. / num_classes), num_samples)
#     return base


#TODO: Update the Below Algorithm description to reflect the new method recommended by Vikash
# Nested Algorithm Development

# 1.) Break into all instances of interest : Label_Extract_Pipeline()
# 2.) Create INDEX of all instances of interests : create_discrete_index()
# --------- For Loop ---------
# 3.) Break INDEX into [(10%) template set| (90%) training/test set]
# 4.) Use INDEX to Break into corresponding [template set| training/test set] : ml_selector()
# 5.) Use template set to make template : ml_selector(make_template=True)
# 6.) Use training/test INDEX and template to create Pearson Features : pearson_extraction()
# 7.) Reorganize into Machine Learning Format : ml_order_pearson()

# Development for Randomized Feature Dropping Analysis Code

def random_feat_drop_analysis(full_trials, all_labels, starts, label_instructions, Class_Obj, offset=int, tr_length=int, k_folds=5,  slide=None, step=False, seed=None, verbose=False):
    """
    ## This needs to be a modular code that will conduct the feature dropping for one feature set
    ## Return (Number of Features (Decreasing or Increasing ?), Number of Nested Folds)

    Parameters:
    -----------
    full_trials: list
        List of Full Epochs, this is typically output from Full_Trial_LFP_Clipper
        [Ch] -> [Freq] -> (Time Samples x Epochs)
    all_labels: list
        List of all Labels corresponding to each Epoch in Full_Trials
        [Epochs]->[Labels]
    starts: list
        List of all Start Times corresponding to each Epoch in Full_Trials
        [Epochs]->[Start Time]
    label_instructions: list
        list of labels and how they should be treated. If you use a nested list in the instructions the labels in
        this nested list will be treated as if they are the same label
    Class_Obj: class
        classifier object from the scikit-learn package
    offset = int
        The number of samples away from the true onset to Grab for ML Trials (Can be Before or After)
    tr_length=int
        Number of Samples to use for Features
    k_folds: int
        Number of Folds to Split between Template | Train/Test sets, defaults to 5,
    slide: bool
        defaults to None
        #TODO: Explain and Validate the Slide Parameter
    step:
        defaults to False
        #TODO: Explain and Validate the Step Parameter
    seed: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is the RandomState instance used by np.random.

    Returns:
    -------
    mean_curve: ndarray

    std_curve: ndarray

    """

    # # Make Array for Holding all of the feature dropping curves
    nested_dropping_curves = [] #np.zeros([])

    # 1.) Break into all instances of interest : label_extract_pipeline()
    clippings, _ = label_extract_pipeline(full_trials=full_trials, all_labels=all_labels, starts=starts[0],
                                          label_instructions=label_instructions, offset=offset, tr_length=tr_length,
                                          Slide=slide, Step=step)

    # 2.) Create INDEX of all instances of interests : create_discrete_index()
    label_identities, label_index = create_discrete_index(clippings, verbose=verbose)
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
        # 4.1) Get template set/training : ml_selector()
        sel_train = ml_selector(clippings=clippings, identity_index=label_identities, sel_instances=X_train,
                                label_index=y_train, make_template=False, verbose=verbose)

        # 4.1) Get test set : ml_selector()
        sel_test = ml_selector(clippings=clippings, identity_index=label_identities, sel_instances=X_test,
                               label_index=y_test, make_template=False, verbose=verbose)

        # 5.) Use template/training set to make template : ml_selector(make_template=True)
        templates = ml_selector(clippings=clippings, identity_index=label_identities, sel_instances=X_train,
                                label_index=y_train, make_template=True, verbose=verbose)

        # 6.1) Use template/training INDEX and template to create Training Pearson Features : pearson_extraction()
        train_pearson_features = pearson_extraction(Clipped_Trials=sel_train, Templates=templates)

        # 6.2) Use test INDEX and template to create Test Pearson Features : pearson_extraction()
        test_pearson_features = pearson_extraction(Clipped_Trials=sel_test, Templates=templates)

        # 7.1) Reorganize Test Set into Machine Learning Format : ml_order_pearson()
        ml_trials_train, ml_labels_train, ordered_index = ml_order_pearson(train_pearson_features)

        # 7.2) Reorganize Training Set into Machine Learning Format : ml_order_pearson()
        ml_trials_test, ml_labels_test, _ = ml_order_pearson(test_pearson_features)

        # 8.) Perform Nested Feature Dropping with K-Fold Cross Validation
        nested_drop_curve = random_feature_dropping(train_set=ml_trials_train, train_labels=ml_labels_train.ravel(),
                                                    test_set=ml_trials_test, test_labels=ml_labels_test.ravel(),
                                                    ordered_index=ordered_index, Class_Obj=Class_Obj, verbose=False)

        nested_dropping_curves.append(nested_drop_curve)

    # 9.) Combine all curve arrays to one array
    all_drop_curves = nested_dropping_curves.pop(0)
    for curve in nested_dropping_curves:
        all_drop_curves = np.concatenate((all_drop_curves, curve), axis=1)

    # 10.) Calculate curve metrics
    mean_curve = np.mean(all_drop_curves, axis=1)
    std_curve = np.std(all_drop_curves, axis=1, ddof=1)  # ddof parameter is set to 1 to return the sample std

    return mean_curve, std_curve



