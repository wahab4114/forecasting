
from config import config
from preprocessing import Preprocessing
from data import Data
from model import Model
from plotting import Plotting



def main():
    # First initiate preprocessor and read files which takes 1 - 2 minutes
    preprocessing = Preprocessing(config['weather_file_path'], config['fire_data_file_path'], ['Datetime'],
                                  ['dt_iso', 'temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg',
                                   'rain_1h', 'rain_3h', 'snow_1h', 'snow_3h', 'clouds_all'])
    # start preprocessing
    preprocessing.start_preprocessing()

    # pass joined weather and fire data to Data class
    data = Data(preprocessing.joined_data)

    # get train test split
    # train data contains 2014 to 2018 years
    # test data contains 2019 as prediction year
    train_x, train_y, test_x, test_y = data.get_train_test()
    xgb = Model((train_x, train_y, test_x, test_y))
    xgb.fit()
    predictions, target = xgb.make_predict()
    description = 'xgb model which contains previous_calls as feature. It exludes weather description feature as it was incearing feature space and was not of importance'
    Plot = Plotting(predictions, target,
                    description,
                    xgb)
    Plot.start_plotting()

if __name__ == '__main__':
    main()