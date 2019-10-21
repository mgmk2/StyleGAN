import argparse
from params import Params
from src.model.stylegan.executor import StyleGAN

parser = argparse.ArgumentParser()
parser.add_argument('--use_tpu', action='store_true')
parser.add_argument('--show_mode', default='pause')
pargs = parser.parse_args()

if __name__ == '__main__':
    p = Params()
    G = StyleGAN(p, use_tpu=pargs.use_tpu, show_mode=pargs.show_mode)
    G.fit()
