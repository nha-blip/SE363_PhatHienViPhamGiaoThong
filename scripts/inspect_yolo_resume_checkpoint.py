import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'weights'
    if not args.checkpoint.resolve().is_relative_to(root.resolve()):
        raise ValueError('Inspect only verified own checkpoints inside workspace weights')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    settings = checkpoint['train_args']
    model = checkpoint.get('ema') or checkpoint.get('model')
    print(json.dumps({'completed_epochs': int(checkpoint['epoch'])+1,
                      'target_epochs': int(settings['epochs']), 'patience': settings.get('patience', 0),
                      'best_epoch': (checkpoint.get('stopper') or {}).get('best_epoch'),
                      'optimizer_present': checkpoint.get('optimizer') is not None,
                      'names': getattr(model, 'names', None)}))


if __name__ == '__main__':
    main()
