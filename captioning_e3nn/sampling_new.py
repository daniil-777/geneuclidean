
import argparse
import config
from sampling.sampler import Sampler
# from utils import Utils



def main():
    parser = argparse.ArgumentParser(
    description='sample from trained model'
)
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    sampler = Sampler(cfg)
    sampler.analysis_cluster()

if __name__ == "__main__":
    main()