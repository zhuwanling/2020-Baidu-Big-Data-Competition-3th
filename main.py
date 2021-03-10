# import chengxun.bosst.run as chengxuanRun
from code188lag45wow8.bosst import run as code188lag45wow8Run
from code78_128.bosst import run as code78_128Run
from code_78_128_55.bosst import run as code_78_128_55Run
from output import fusion
from lstm.lstm import run as lstmRun
from lstm.output import run as lstmoutputRun




if __name__ == "__main__":
    # chengxuanRun()
    code188lag45wow8Run()
    code78_128Run()
    code_78_128_55Run()
    fusion()
    print("添加LSTM权重")
    lstmRun()
    lstmoutputRun()

