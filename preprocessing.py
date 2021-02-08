import pandas as pd
from config import config

# this class does preprocessing and feature engineering of the provided "Seattle_Real_Time_Fire_911_Calls.csv" and "Seattle Weatherdata 2002 to 2020.csv"
# file1: path to first file
# file2: path to second file
# feature_set_1: list of the columns to read from file 1
# feature_set_2: list of the columns to read fomr file 2
# usage: call start_preprocessing function and use joined_data after that
class Preprocessing():
    def __init__(self, file1, file2, features_set_1, features_set_2):
        self.file1 = file1
        self.file2 = file2
        self.features_set_1 = features_set_1
        self.features_set_2 = features_set_2
        print('reading_files_start')
        self.fire_data = pd.read_csv(file1, parse_dates=['Datetime'], usecols=features_set_1)
        self.weather_data = pd.read_csv(file2, usecols=features_set_2)
        print('reading_files_end')

    def start_preprocessing(self):
        # here order of the function calls matter
        print('--Preprocessing start--')
        self.preprocess_weather_data()
        self.format_datetime()
        self.pruning_data()
        self.make_target_feature()
        self.merging_data()
        self.split_date_time()

    def preprocess_weather_data(self):
        # most of the seasonal data (rain, wind, snow etc) contains empty values, but it is of use
        # therefore empty/null values can be replaced with 0
        self.weather_data = self.weather_data.fillna(0)

    #         one_hot_encoding = pd.get_dummies(self.weather_data['weather_description'], prefix='weather')
    #         self.weather_data = pd.concat([self.weather_data, one_hot_encoding], axis=1)
    #         self.weather_data = self.weather_data.drop(columns = ['weather_description'])

    def format_datetime(self):
        print('--format datetime--')
        # format datetime to make it standard
        # in case of fire_data we want to use hourly information and discard information on minutes level
        self.fire_data['Datetime'] = self.fire_data['Datetime'].dt.strftime('%Y-%m-%d %H')
        # remove extra UTC info
        self.weather_data['dt_iso'] = self.weather_data['dt_iso'].str.replace('\+0000 UTC', '')
        # convert to datetime object and format
        self.weather_data['dt_iso'] = pd.to_datetime(self.weather_data['dt_iso'])
        self.weather_data['dt_iso'] = self.weather_data['dt_iso'].dt.strftime('%Y-%m-%d %H')

    def pruning_data(self):
        print('--data pruning--')
        # weather data is not updated and contains values till date 2020-11-05 00
        # therefore for simplicity I will use test year as 2019 and last five years starting from 2014 as train
        self.weather_data = self.weather_data[
            (self.weather_data['dt_iso'] >= '2014-01-01') & (self.weather_data['dt_iso'] < '2020-01-01')]
        self.fire_data = self.fire_data[
            (self.fire_data['Datetime'] >= '2014-01-01') & (self.fire_data['Datetime'] < '2020-01-01')]

    def make_target_feature(self):
        print('--make target feature--')
        # Group hourly data to get the count of calls per hour which will be our target feature
        # we can say that we are transforming data into supervised learning problem
        self.fire_data = pd.DataFrame(self.fire_data.groupby('Datetime')['Datetime'].count()).rename(
            columns={'Datetime': 'calls'}).reset_index()
        # weather data contains duplicate values of dates
        self.weather_data = self.weather_data.drop_duplicates(subset=['dt_iso'])

    def merging_data(self):
        print('--merging data--')
        ##checking if weather data contains each and every hour of every day in the time period by comparing length
        # date_rng = pd.DataFrame(pd.date_range(start='1/1/2014', end='31/12/2019', freq='H'))
        # date_rng.rename(columns={0:'Time'}, inplace=True)
        # Left-join of weather_data and fire_data in order to get complete features (whole data)
        self.joined_data = self.weather_data.merge(self.fire_data, how='left', left_on='dt_iso', right_on='Datetime')
        # filling null values with zeros
        self.joined_data = self.joined_data.fillna(0)
        self.joined_data.drop(columns=['Datetime'], inplace=True)
        self.joined_data = self.joined_data.sort_values(by='dt_iso')

    def split_date_time(self):
        print('--split date time--')
        # splitting datetime column into multiple column so that data information at individual level can be processed
        # by model
        self.joined_data['dt_iso'] = pd.to_datetime(self.joined_data['dt_iso'])
        self.joined_data['hour'] = self.joined_data['dt_iso'].dt.hour
        self.joined_data['dayofweek'] = self.joined_data['dt_iso'].dt.dayofweek
        self.joined_data['month'] = self.joined_data['dt_iso'].dt.month
        self.joined_data['year'] = self.joined_data['dt_iso'].dt.year
        self.joined_data['dayofyear'] = self.joined_data['dt_iso'].dt.dayofyear
        self.joined_data['dayofmonth'] = self.joined_data['dt_iso'].dt.day
        self.joined_data.drop(columns=['dt_iso'], inplace=True)




def main():
    ## dummy main for testing
    ## this code will be used in main
    preprocessing = Preprocessing(config['weather_file_path'], config['fire_data_file_path'], ['Datetime'],
                                  ['dt_iso', 'temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg',
                                  'rain_1h', 'rain_3h', 'snow_1h', 'snow_3h', 'clouds_all'])
    preprocessing.start_preprocessing()
    print(preprocessing.joined_data)
if __name__ == '__main__':
    main()