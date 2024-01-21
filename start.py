from mvector.utils.utils import dict_to_object
from ruamel import yaml

from loadTXT import loadTXT
from predict import predict
from train import train
from eval_LFW import evalLFW

# import matplotlib
# matplotlib.use('TkAgg')
if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf-8') as f:
        configs = yaml.load(f.read(), Loader=yaml.RoundTripLoader)
    config = dict_to_object(configs)
    if config.start.reloadData:
        loadTXT(config=config.dataset)
    if config.start.startTrain:
        train(config=config.train, lfw=config.LFW)
    if config.start.evalLFW:
        evalLFW(config=config.LFW)
    if config.start.predict:
        predict(config=config.predict)
    pass
