
from matplotlib import pyplot
import numpy as np
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import config
from preprocessing import Preprocessing
from data import Data
from model import Model

class Plotting:
    def __init__(self, prediction, target, description, xgb):
        self.prediction = prediction
        self.target = target
        self.description = description
        self.xgb = xgb

    def start_plotting(self):
        self.print_errors()
        self.plot_feature_importance()
        self.plot_back_testing()

    def print_errors(self):
        ## Plotting around 8k samples is not going to be useful visually
        ## We can plot results of each day instead of the hours
        self.preds_day = np.add.reduceat(self.prediction, np.arange(0, len(self.prediction), 24))
        self.y_day = np.add.reduceat(self.target, np.arange(0, len(self.target), 24))
        ## calculating RMSE per hour
        print("RMSE per hour", mean_squared_error(self.target, self.prediction, squared=False))
        ## print RMSE day wise
        print("RMSE per day", mean_squared_error(self.y_day, self.preds_day, squared=False))

    ## plotting the importance of features
    def plot_feature_importance(self):
        print(self.description)
        fig, ax = pyplot.subplots(figsize=(10, 8))
        _ = plot_importance(self.xgb.model, height=0.9, ax=ax)
        pyplot.savefig(config['feature_importance_image_path'])
        pyplot.show()

    def plot_back_testing(self):
        pyplot.figure(figsize=(15, 5))
        pyplot.title("prediction vs target")
        pyplot.plot(self.y_day, label='target')
        pyplot.plot(self.preds_day, label='prediction')
        pyplot.legend()
        pyplot.savefig(config['backtesting_image_path'])
        pyplot.show()


def main():
    ## dummy test for code working till plotting class
    preprocessing = Preprocessing(config['weather_file_path'], config['fire_data_file_path'], ['Datetime'],
                                  ['dt_iso', 'temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg',
                                   'rain_1h', 'rain_3h', 'snow_1h', 'snow_3h', 'clouds_all'])
    preprocessing.start_preprocessing()
    data = Data(preprocessing.joined_data)
    train_x, train_y, test_x, test_y = data.get_train_test()
    xgb = Model((train_x, train_y, test_x, test_y))
    xgb.fit()
    predictions, target = xgb.make_predict()
    Plot = Plotting(predictions, target,
                    'xgb model which contains previous_calls as features. It exludes weather description feature as it was incearing feature space and was not of importance',
                    xgb)
    Plot.start_plotting()
if __name__ == '__main__':
    main()