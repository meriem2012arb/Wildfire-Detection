# Wildfire Detection 

# Project Description 

    
### Goal : 
Our objective is to differentiate between an image that shows forest with fire from an image of forrest without fire, using Transfer Learning
 
![alt text](https://github.com/meriem2012arb/Capstone1_project/blob/main/images/006.jpg)


This project is a part of the [ml-zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) course

Data :
The  Original dataset dowloaded is  from [Kaggle](https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data)

### Information about dataset :
```
folders: 
 forest_fire :
      Testing :1832 images belonging to 2 classes.
      Training and Validation : 68 images belonging to 2 classes.
      
      Classe : 'Fire': 0, 'No Fire': 1
 ```
### Table of Contents :

```images``` Folder contains some images (not from the used dataset ) 

```Dockerfile``` of a Docker container

```Pipfile```  for the virtual environment using pipenv

```Pipfile.lock```  for the virtual environment using pipenv

```notebook.ipynb``` Jupyter notebook with the Exploratory Data Analysis and Model selection

```predict.py``` Model loading and web service deployment

```predict_test.py``` Output testing locally

```requirements.txt``` requirements file for ```predict.py``` deployment

```train.py``` Final model training (saved model) 


### Model training

3 architectures were chosen for Transfer Learning:

```
Xception
ResNet
Inception
```




### Run the Code


Pipenv creates an enviroment with the name of the current folder.

Install 'pipenv' running in shell:

```pip install pipenv```

Activate the environment running in shell:

```pipenv shell```

When then environment is activated, install everything using 'pipenv' instead of 'pip', for this project, to creat the Pipfile and the Piplock.file, we run (since you have them already in the folder, you do not need to run the following command line):

``` pipenv install  tensorflow keras pillow flask gunicorn``` 

The Pipfile records what you have installed (thus only run the packages installation once) and in the Pipfile.lock are the packages checksums.

Close the environment with Crtl + d

To use the environment, run ```pipenv shell``` and deploy the model as said in the next section.

Run ```train.py``` for train and save the model. 

```python3 train.py```

##  Apply the deployment

In the active environment and open the web server by running:

```gunicorn --bind 0.0.0.0:9696 predict:app```

(use 'waitress' instead of 'gunicorn' if you are in Windows).

 image from ```images``` is imported in 'predict_test.py'. Test the deployment by running it in other shell:

```python3 predict_test.py```

The output (if that image shows fire forest or not and the probability) will be written in the shell.

Close the web server with Ctrl + c.

##  Docker

We do not need to install packages, activate environments, train models,... everytime we want to know if a new image shows fire forest  or not. We can skip the former sections using a Docker container.

First, create a Docker image locally by running in shell (the enviroment does not need to be activated):

```docker run -it --rm --entrypoint=bash python:3.8.12-slim```

Exit the container shell with Ctrl + d.

The Dockerfile is this folder installs python, runs pipenv to install packages and dependencies, runs the predict.py script to open the web server and the  model and deploys it using gunicorn

```docker built -t docker-fireforest .```
(the last point means 'here', i.e., run it in the environment folder).

Run the docker container with:

```docker run -it --rm -p 9696:9696 docker-fireforest```

and the model will be deployed and ready to use.

To send a new request, open a new shell in the enviroment directory and directly run:

```python3 predict_test.py```

and you will see if the image shows fire forest  or not and its probability.

Close the container with Ctrl + c.


