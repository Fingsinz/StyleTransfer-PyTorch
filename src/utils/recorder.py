import time
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

    def log(self, text: str, end='\n') -> None:
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = '[' + now_time + ']' + '\t' + text
        
        with open(self.filename, 'a') as f:
            f.write(text)
            f.write(end)

        print(text, end=end)
