import xgboost as xgb
import tqdm
from config import config
from preprocessing import Preprocessing
from data import Data

## this class does training and prediction apart from model initialization
class Model:
    def __init__(self,data):
        self.train_x, self.train_y, self.test_x, self.test_y = data
        self.model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators = 1000)
    def fit(self):
        print('--training started--')
        self.model.fit(self.train_x, self.train_y)
        print('--training finished--')
    ## "previous_hour_calls" is one of the generated feaute which tells number of the calls in previous hour
    ## test data does not contain this "previous_hour_calls" feature
    ## Therefore it has to be calculated on the go. I used calls of the last training example to set "previous_hour_calls" for the first test example
    ## For later test example we used prediction of the previous test example to set "previous_hour_calls"
    def make_predict(self):
        print('--prediction started--')
        predictions = []
        originals = []
        pred = int(self.train_y.iloc[-1].values) # calls of the last training example
        for i in tqdm.tqdm(range(len(self.test_x))):
            x = self.test_x[i:i+1]
            x = x.assign(previous_hour_calls = pred) # update "previous_hour_calls" for current test example
            y = self.test_y[i:i+1]
            pred = self.model.predict(x)
            predictions.append(round(pred[0])) # model returns list of one element
            originals.append(int(y.values))
        return predictions, originals

def main():
    ## dummy test for code working till model class
    preprocessing = Preprocessing(config['weather_file_path'], config['fire_data_file_path'], ['Datetime'],
                                  ['dt_iso', 'temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg',
                                   'rain_1h', 'rain_3h', 'snow_1h', 'snow_3h', 'clouds_all'])
    preprocessing.start_preprocessing()
    data = Data(preprocessing.joined_data)
    train_x, train_y, test_x, test_y = data.get_train_test()
    xgb = Model((train_x, train_y, test_x, test_y))
    xgb.fit()
    predictions, target = xgb.make_predict()

if __name__ == '__main__':
    main()