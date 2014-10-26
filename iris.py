#!/usr/bin/env python

import sklearn.datasets
import sklearn.ensemble

import pandas
import numpy


# pick half of the data, randomly, returning the two halves as a tuple.
def split_data_randomly(dataframe):
    half_point_count = len(dataframe) // 2
    point_array = numpy.random.permutation(range(len(dataframe)))
    indices_1 = point_array[:half_point_count]
    indices_2 = point_array[half_point_count:]
    return dataframe.iloc[indices_1], dataframe.iloc[indices_2]


def main():
    print('loading...')
    iris = sklearn.datasets.load_iris()
    dataframe = pandas.DataFrame(iris.data, columns=iris.feature_names)
    dataframe['species'] = pandas.Categorical(iris.target, iris.target_names)

    # use half of the data points for training, and try to predict the rest.
    train_data, predict_data = split_data_randomly(dataframe)

    print('training...')
    classifier = sklearn.ensemble.RandomForestClassifier()
    classifier.fit(train_data[iris.feature_names], train_data['species'])

    print('predicting...')
    predictions = classifier.predict(predict_data[iris.feature_names])
    r = pandas.crosstab(predict_data['species'], predictions,
                        rownames=['actual'], colnames=['preds'])
    print(r)


if __name__ == '__main__':
    main()
