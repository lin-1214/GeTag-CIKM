import os
from argparse import ArgumentParser
import naive_flow as nf
from .configs import ExperimentConfig


def main():
    arg_parser = ArgumentParser('llm-tag-downstream')
    arg_parser.add_argument(
        '-e',
        '--env',
        type=str,
        required=True,
        help='path to the .env file to use as config',
    )
    arg_parser.add_argument(
        '--check',
        action='store_true',
    )
    arg_parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
    )
    args = arg_parser.parse_args()
    env_path = args.env
    assert os.path.isfile(env_path), 'No env file found'

    data = nf.load_env_file(env_path, preset_env_vars={'__file__': env_path})
    config = ExperimentConfig.model_validate_strings(data)
    print(nf.strfconfig(config))
    if args.check:
        return

    results = config.eval(cache_dir='exps', verbose=args.verbose)

    from pprint import pprint
    pprint(results)


if __name__ == '__main__':
    main()
