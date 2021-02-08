
import pandas as pd
from config import config
from preprocessing import Preprocessing

# pass joined data
# data: dataframe of merge/join of both fire and weather data
class Data:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_train_test(self):
        # splitting into train and test data
        # train_data consists of data from year 2014 to 2018
        train_data = self.data[self.data['year'] < 2019]
        # test_data consists of year 2019
        test_data = self.data[self.data['year'] == 2019]

        # add another colum which holds the number of calls of previous point in time (hour)

        train_data = train_data.assign(previous_hour_calls=train_data['calls'].shift(1))
        train_data = train_data.dropna()
        # separate train_x and train_y
        train_y = pd.DataFrame(train_data['calls'])
        train_x = train_data.drop(columns=['calls'])

        # separate test_x and test_y
        test_y = pd.DataFrame(test_data['calls'])
        test_x = test_data.drop(columns=['calls'])
        return train_x, train_y, test_x, test_y


def main():
    ## dummy test for code working till data class
    preprocessing = Preprocessing(config['weather_file_path'], config['fire_data_file_path'], ['Datetime'],
                                  ['dt_iso', 'temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg',
                                   'rain_1h', 'rain_3h', 'snow_1h', 'snow_3h', 'clouds_all'])
    preprocessing.start_preprocessing()
    data = Data(preprocessing.joined_data)
    train_x, train_y, test_x, test_y = data.get_train_test()
    print(train_x)
if __name__ == '__main__':
    main()