# CS50-AI-P5-traffic
Project for lecture 5 Neural Networks to "Artificial Intelligence with Python" Harvard course

The aim of this project is to create an AI which identifies which traffic sign appears in a photograph.
It uses the gtsrb dataset. Experimenting with different neural network architectures, the best model achieved 98% accuracy on the test set.

# Model architectures and results
## Model 1
This is a quite simple model with only 2 layers - a flatten layer and a dense layer with softmax activation function. It is fairly fast. Accuracy on the test set is 83%. Accuracy on the training set ranged a lot, but best is 82%.
```
model = tf.keras.models.Sequential(
    tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
)
```

## Model 2
Second model tried has 3 layers - a flatten layer, a dense layer with 128 units which uses relu activation function and a dense layer with softmax activation function. It is a bit slower than the first model. It performed very poorly on both training and test sets. On the test set it achieved 5% accuracy. On training set first epoch had accuracy of 4% with loss between 14 and 20. Afterwards the loss dropped to 3.6. Minimum loss achived was 3.5 with accuracy 5%, sometimes 6%.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 3
Instead of directly flattening the image with Model 3 we use a convolutional layer with 16 filters. It is followed by a flatten layer and a dense layer with softmax activation function. It is slower than the previous models. On training set the accuracy started at 58% and rose to 97%. Loss went from 19.5 to 0.16.
On the test set this model achieved 91% accuracy with loss 1.25.

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 4
Model 4 is similar to Model 3, but instead of 16 filters it uses 32 filters in the convolutional layer. It is slower than Model 3. On the training set accuracy started at 67%, but loss was 25.93. After 10 epochs accuracy was 96% and loss 0.19. On the test set this model achieved 90% accuracy with loss 1.15.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 5
Model 5 uses 32 filters on its convolutional layer, but their size is 7x7. What was observed is that the loss started out smaller than any of the other models - 2.93, but accuracy was 50%. After 10 epochs accuracy was 92% and loss 0.33. On the test set this model achieved 86% accuracy with loss 1.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (7, 7), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 6
To avoid overfitting, with Model 6 we add a dropout of 50%. Also, the number of filters used is 64, small size 3x3. This model is noticeably slower than the previous ones. On the training set accuracy started at 60% and loss at almost 23. After 10 epochs accuracy was 92% and loss 0.44. On the test set this model achieved 92% accuracy with loss 0.85.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 7
For the next model we use Model 6 as base, but add max-pooling layer with pool size 3x3. What pooling is reducing the size of the image by taking the maximum value in a given area. On the training set accuracy started at 43%, but on the second epoch rose to 71%. Loss started at 4.3, but dropped to 1.16. After 10 epochs accuracy was 83% and loss 0.7. On the test set this model achieved accuracy 90% with loss 0.4.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 8
Model 8 is the same as Model 7, but the pool size is 2x2. First epoch on the training set had accuracy of 55% with loss 7. After 10 epochs accuracy was 89% and loss 0.5. On the test set this model achieved accuracy 93% with loss 0.4, which makes it the best model so far.

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 9
This model is two times Model 8, meaning we use 2 convolutional layers each followed by a max pooling 2x2 layer. The filters, their size and dropout remain the same. This model achieved accuracy of 89% with loss of 0.4 on the training set. On the test set this model achieved accuracy 95% with loss 0.2, which makes it the best model so far.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 10
For Model 10, we add a new dense layer with 128 units and relu activation function. This model was slower than the previous one. Accuracy achieved on training set is 91% with loss 0.3. On the test set this model achieved accuracy 95% with loss 0.2, which makes the same output as Model 9.
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])
```

## Model 11
Model 11 uses 2 convolutional layers with relu activation function - one with 64 filters 3x3 and second with 128 filters of 2x2 size. Each convolutional layer is followed by a max pooling 2x2 filter. Dropout is reduced to 0.3. This model was faster than the previous one. Accuracy achieved on training set is 96% with loss 0.15. On the test set this model achieved accuracy 98% with loss 0.09, which makes the model with the best results so far.
```
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (2, 2), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

## Overall results
Best model is Model 11, which achieved 98% accuracy on the test set. It is faster than some of the other models. Loss is also very low. Other well-performing models are Model 9 and Model 10, which achieved 95% accuracy on the test set.

Worst model turns out to be Model 2, which achieved 5% accuracy on the test set, even though the loss was low.


## Important note
- All of these models were trained with 10 epochs.
- The data set was split into 40% validation set and 60% training set.
- Each model was trained with Adam optimizer and sparse categorical crossentropy loss function.
- Results are averaged over runs.
```
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

## How to use the project
To run this project, you will need the ```scikit-learn```, ```opencv-python``` and ```tensorflow``` python packages. Make sure to install the requirements using the following command in the root folder:
```
pip3 install -r requirements.txt
```
if not already installed.


How to run:
```
python3 traffic.py data_directory [model.h5]
```
- first argument is the path to the directory containing the data set
- second argument is optional, it is filename where the trained model will be saved

