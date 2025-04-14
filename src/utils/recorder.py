from utils.utils import check_dir

class Recorder:
    def __init__(self, output_path: str, output_file: str) -> None:
        self.filename = check_dir(output_path) + output_file

    def set_statistic(self, statistic: [str]) -> None:
        self.statistic = statistic
        with open(self.filename, 'w') as f:
            f.write(','.join(statistic))
            f.write('\n')

    def record(self, text: [str]) -> None:
        with open(self.filename, 'a') as f:
            f.write(','.join(text))
            f.write('\n')

