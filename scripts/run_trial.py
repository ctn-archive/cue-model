from pytry import parser

from cue.model.trial import CueTrial


trial = CueTrial()

if __name__ == '__main__':
    args = parser.parse_args(trial, args=None, allow_filename=False)
    trial.run(**args)
