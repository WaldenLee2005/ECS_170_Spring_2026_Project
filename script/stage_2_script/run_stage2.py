import os
import matplotlib.pyplot as plt

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Setting_Train_Test_Fixed import Setting_Train_Test_Fixed
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_1_code.Result_Saver import Result_Saver


def main():
    # dataset
    data_obj = Dataset_Loader('stage 2 dataset', 'pre-split train/test csv files')
    data_obj.train_file_path = './data/stage_2_data/train.csv'
    data_obj.test_file_path = './data/stage_2_data/test.csv'

    # method
    method_obj = Method_MLP('mlp', 'PyTorch MLP for multiclass classification')

    # result saver
    result_obj = Result_Saver('result', 'save experiment results')
    result_obj.result_destination_folder_path = './results/'
    result_obj.result_destination_file_name = 'stage2_mlp_result'

    # evaluator
    evaluate_obj = Evaluate_Metrics('metrics', 'accuracy/precision/recall/f1')

    # setting
    setting_obj = Setting_Train_Test_Fixed('train/test fixed', 'use instructor-provided train/test split')
    setting_obj.dataset = data_obj
    setting_obj.method = method_obj
    setting_obj.result = result_obj
    setting_obj.evaluate = evaluate_obj

    os.makedirs('./results/', exist_ok=True)
    os.makedirs('./plots/', exist_ok=True)

    metrics, _ = setting_obj.load_run_save_evaluate()

    # draw convergence plot
    loss_history = method_obj.loss_history
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('MLP Training Convergence Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/stage2_mlp_convergence.png')
    plt.show()

    print('\nFinal metrics dictionary:')
    print(metrics)


if __name__ == '__main__':
    main()