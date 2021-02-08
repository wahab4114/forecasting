# Seattle Real Time Fire 911 Calls
Forecasting of number of emergency calls for Seattle Fire Department

#### Feature engineering
Both "Seattle_Real_Time_Fire_911_Calls.csv" and "Seattle Weatherdata 2002 to 2020.csv" contain many features.
I selected few of them based on importance and later I included few which were not selected at the first place e.g. weather desciption.
I also added another feature which is called "previous_hour_calls" which contains calls of the previous hours as a feature in the training data.
Because of this, I did testing in a linear manner where predictions were being added into the next test example as a feature, on the go.

#### About Design / Stucture
* notebook/ contains one .ipynb file which contains whole project in one sequence
* notebook/ contains run_1.jpg, run_2.jpg and run_3.jpg:
  * run_1.jpg contains result of the model when I added another one hot encoded feature called 'weather description'
  * run_2.jpg contains result of the model without "previous_hour_calls" feature. It decreased the model perfomance.
    which shows the importance of this feature
  * run_3.jpg conatins result of the model where I kept "previous_hour_calls" (current code) as a feature and it increased the performance.
  
* All other .py files belongs to the module code which I ran using Pycharm and contains the best performing model
* results/ contains result of the .py code

#### To run
* you need to have Anaconda installed
* Run commad "conda env create -f nl.yml" to create environment from provided .yml file. Name of the env is 'nl'
* After this, activate the conda environment
* To run .ipynb, you need to set the path for both files at the beginning of the file and you can run
* To run .py code, you to need to set path for both data files in the config.py. You also need to set path (to save) for feature importance plot as well as backtesting plot
* After setting the environment and paths, you can simply run main.py and it should work

#### Data
* Used data of 5 years from 2014 to 2018 for training the model and year 2019 for testing
  because weather info for year 2020's last two month was missing
* I did training on hour level, which means it includes year, month, day and hour.

#### Model and evaluation
* I used simple XGBoost model with objective reg:squared_error, becuase it is a regression problem
* Used Root Mean Squared Error (RMSE) to measure the performance of the model
  * RMSE is calculated daywise as well as hourly
* Backtesting plot is plotted daywise instead of plotting around 8k hourly predictions for visual simplicity.

#### dependencies
Creating conda env from yml file should not be a problem but here is the list of modules/dependencies needed to be in the environment in case of any issue
with creating env from yml file
* Python 3.6.12 (python version)
* pip install pandas
* pip install matplotlib
* pip install scikit-learn
* pip install xgboost (sklearn needs to be installed before xgboost)
* pip install tqdm


Note: all of the files contain commenting.

