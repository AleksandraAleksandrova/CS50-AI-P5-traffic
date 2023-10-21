# CS50-AI-P5-traffic
Project for lecture 5 Neural Networks to "Artificial Intelligence with Python" Harvard course

The aim of this project is to create an AI which identifies which traffic sign appears in a photograph.
It uses the gtsrb dataset.


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

