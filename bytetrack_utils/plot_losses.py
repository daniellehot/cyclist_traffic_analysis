import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("-i", "--input", type=str, help="Metrics json")
    return parser.parse_args()


def plot_losses(input):
    data_all = pd.read_json(input, lines=True) 
    epochs = data_all['epoch'].unique()
    epochs = np.append(epochs, -1)
    
    for epoch in epochs:
        if epoch != -1:
            print(f"Plotting epoch {epoch}")
            data = data_all.loc[data_all['epoch'] == epoch]
            output = f"epoch_{epoch}.png"
        else:
            print("Plotting trainning summary")
            data = data_all
            output = "losses_summary.png"

        fig, axes = plt.subplots(5, 1, figsize=(10, 20))
        losses = ["total_loss", "iou_loss", "l1_loss", "conf_loss", "cls_loss"]
        titles = ["Total Loss", "IoU Loss", "L1 Loss", "Confidence Loss", "Classification Loss"]

        for idx, loss in enumerate(losses):
            if epoch != -1:
                axes[idx].plot(data['iter'], data[loss], label=titles[idx])
            else:
                number_of_entries = np.arange(data.shape[0])
                axes[idx].plot(number_of_entries, data[loss], label=titles[idx])
                axes[idx].set_xticks(number_of_entries[::len(data) // max(data['epoch'])])
                axes[idx].set_xticklabels(data['epoch'].unique())
            
            axes[idx].set_title(titles[idx])
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].legend()

        # Adjust layout
        plt.tight_layout()

        output = copy.deepcopy(input).replace("metrics.json", output)
        # Save the figure
        plt.savefig(output)

if __name__=="__main__":
    plot_losses(parse_args().input)