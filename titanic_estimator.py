# pylint: disable=unused-import
# pylint: disable=missing-docstring
# pylint: disable=trailing-newlines
import tensorflow as tf
#import titanic_data


def feature_columns():

    # each passenger has a unique ID so this feature should be categorical
    passenger_id = tf.feature_column.categorical_column_with_identity(
        key='PassengerId',
        num_buckets=1310
    )
    # as of the time of writing this, hashed categorical columns
    # are not supported in DNNClassifier Estimator
    # we have to use either indicator column or embedding column to wrap our feature in
    passenger_id = tf.feature_column.indicator_column(passenger_id)

    # 667 unique values in family feature so we use hashed column
    family = tf.feature_column.categorical_column_with_hash_bucket(
        key='Family',
        hash_bucket_size=667
    )
    family = tf.feature_column.indicator_column(family)

    # pclass is a multiclass low dimension feature
    # so IdentityCategoricalColumn would be the best option
    p_class = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Pclass',
        vocabulary_list=[1, 2, 3])
    p_class = tf.feature_column.indicator_column(p_class)

    # two class feature with vocabulary list
    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Sex',
        vocabulary_list=['male', 'female'])
    sex = tf.feature_column.indicator_column(sex)

    # age is obviously a numeric value but in our problem
    # it's better to bucketize it in a way it is more affective in training
    numeric_age = tf.feature_column.numeric_column(
        key='Age',
        dtype=tf.int8)
    age = tf.feature_column.bucketized_column(
        source_column=numeric_age,
        boundaries=[12, 20, 40, 60, 80]
    )
    # a simple numeric column
    family_members = tf.feature_column.numeric_column(
        key='FamilyMembers',
        dtype=tf.int64
    )
    embarked = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Embarked',
        vocabulary_list=['S', 'C', 'Q']
    )
    embarked = tf.feature_column.indicator_column(embarked)
    # again a low dimensional categorical value
    deck = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Deck',
        vocabulary_list=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'U']
    )
    deck = tf.feature_column.indicator_column(deck)

    columns = [passenger_id, family, p_class, sex, family_members, age, embarked, deck]

    return columns


def input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


#pylint: disable=invalid-name
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns(),
    hidden_units=[20, 20, 20, 20, 20, 20],
    n_classes=2,
    model_dir="C:\\Users\\Aryan\\Source\\Kaggle\\titanic\\titanic_model\\test",
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.05,
        l1_regularization_strength=0.001,
        l2_regularization_strength=0.001
    )
)
