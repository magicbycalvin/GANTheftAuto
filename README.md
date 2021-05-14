# GANTheftAuto 

The purpose of this project is to generate optimal local trajectories for a self driving vehicle equipped with a front-facing camera. The trajectories are generated using a generative adversarial network (GAN) and feasibility checks are performed using an additional, specialized discriminator. The original purpose of this project is for the course ECE 5995. Data and Trajgan are the main folders required for this project. 

 

# Data 

The data folder contains all the raw camera feed recorded from Carla as well as the perspective transform images. The preprocessed images are then copied into the training folder of Trajgan for training the generator.  

 

## Carla-Recordings 

This folder contains the raw camera images from various Carla recordings. They are labeled in separate folders with the out files being straight only paths, the long_out01 being a long recording from the dummy controller in Town01, and the out_0# being long recordings from Town06. All recordings are of the built in Carla semantic image segmentation of 640x480 images.  

## PreProc 

The PreProc folder contains the transformed data. The processed data is cropped into a 480x480 image and transformed into a 2D top view to feed into the generator 

 

# Trajgan 

Trajgan is the main folder where all the code for running the project is contained. It is separated into four folders, control, gan, path_gen, and perception. 

## Control 

The control folder contains the code necessary to run simulations in Carla. Carla can be downloaded for Windows or Linux at https://carla.readthedocs.io/en/0.9.11/start_quickstart/ note that version 0.9.11 must be used. The main program is gan_control.py which runs the client to communicate with the Carla Server and operates the remaining programs in the control file. The program currently will spawn a Tesla Model 3, generate a global path to follow and navigate that path using waypoints that are spaced roughly every 7 meters. The program will end once the car reaches the destination. Keyboard controls available are as follows: Key’s 0-9 control which sensor is active, the image segmentation can be viewed by pressing 6. R toggles the current sensor recording on or off, images will be saved in the parent directory to a folder called _out, and q aborts the program. To run the program first the Carla server must be started by running ./CarlaUE4.sh and then in a separate terminal ./gan_control.py can be run. 

 

## GAN 

This folder contains several versions of the GAN implemented 

### training 

The training folder contains three different Jupyter Notebook files and a training_data folder. The three. ipynb files were created to program the GAN, in which GANNAV_Restructure_1.ipynb is the newest version of the GAN implementation, and data preprocessing.  

### training_data 

The training_data folder contains the train_data folder and the dataset.csv file. The csv file is used as an input for the GAN, this file has the necessary x and y desired control points for each image.  

### train_data 

This folder has all the data images after segmentation used to train the GAN. 

## path_gen 

This folder contains 2 folders, “optimization” and “polynomial”, and path_generator.py file.  
 
