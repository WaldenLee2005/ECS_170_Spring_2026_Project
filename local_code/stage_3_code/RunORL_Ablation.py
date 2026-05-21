import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from local_code.stage_3_code.Method_CNN_ORL_Ablation import Method_CNN_ORL

def main():
    # load data
    with open('./data/stage_3_data/ORL', 'rb') as f:
        data = pickle.load(f)

    # method
    method_obj = Method_CNN_ORL('cnn_orl', 'CNN for ORL')
    method_obj.data = data

    os.makedirs('./results/', exist_ok=True)
    os.makedirs('./plots/', exist_ok=True)

    # run
    result = method_obj.run()
    pred_y = result['pred_y']
    true_y = result['true_y']
    loss_history = result['loss_history']

    # learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('CNN ORL Training Convergence Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/stage3_cnn_orl_20_epochs_convergence.png')
    plt.show()

    # metrics
    acc  = accuracy_score(true_y, pred_y)
    prec = precision_score(true_y, pred_y, average='macro')
    rec  = recall_score(true_y, pred_y, average='macro')
    f1   = f1_score(true_y, pred_y, average='macro')

    print('\n--- Evaluation Results ---')
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1 Score:  {f1:.4f}')

if __name__ == '__main__':
    main()