import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PARSED_METRIC_PATH = r"../SC#01_N_W1_LC_R1/trace/metric_parsed/log_power_30_demo_core1_iter1_parsed.csv"


LABEL_FONT_SIZE = 16
WIDE_FIG_SIZE = (12, 6)
PARAM_FONT_SIZE = 14
Y_LIMIT = (0.61, 1)


class ProcessCSV:
    """
    This class is responsible for handling the CSV that contains the stored metrics.
    It also displays collected metrics to scatter plots.
    """

    _OUTPUT_FILE = r"..\output"

    def __init__(self):
        df = pd.read_csv(PARSED_METRIC_PATH)
        self.df = df
        self.path = PARSED_METRIC_PATH

        cycles = df["cycle_id"].to_numpy()
        cycle_ids = np.unique(cycles)
        self.cycle_indexes = [np.where(cycles == uid)[0] for uid in cycle_ids]
        self.timestamp = df["timestamp"].to_numpy()
        self.image_op_flag = df["image_op_flag"].to_numpy()

        self.image_metric, self.image_interval = self.get_image_cycles()
        self.neural_metric, self.neural_interval = self.get_neural_cycles()

    def get_all_cycles(self) -> tuple[list, list]:
        cycle_timestamp = []
        cycle_metric = []
        metric = self.df["current-amps"].to_numpy()
        for cycle in self.cycle_indexes:
            cycle_metric.append(metric[cycle])
            cycle_timestamp.append(self.timestamp[cycle])

        return cycle_metric, cycle_timestamp

    def get_image_cycles(self) -> tuple[list, list]:
        timestamps_image = []
        metric_image = []
        metric = self.df["current-amps"].to_numpy()
        for cycle in self.cycle_indexes:
            image_mask = self.image_op_flag[cycle]
            timestamp_cycle = self.timestamp[cycle]
            intersection = np.where(image_mask == 1)
            timestamps_image.append(timestamp_cycle[intersection])
            metric_image.append(metric[intersection])

        return metric_image, timestamps_image

    def get_neural_cycles(self) -> (np.ndarray, np.ndarray):
        cycles, time = self.get_all_cycles()
        image, _ = self.get_image_cycles()
        neural, neural_time = [], []
        for c, t, i in zip(cycles, time, image):
            n = np.array(c[len(i):], copy=True)
            n_t = np.array(t[len(i):], copy=True)
            neural.append(n)
            neural_time.append(n_t)
        return neural, neural_time

    def scatter_neural(self) -> None:
        for n, t in zip(self.neural_metric, self.neural_interval):
            plt.figure(figsize=WIDE_FIG_SIZE)
            plt.scatter(t, n, s=3, label="Electrical current trace")
            plt.xlabel("Time in seconds", fontsize=LABEL_FONT_SIZE)
            plt.ylabel("Electrical current", fontsize=LABEL_FONT_SIZE)
            plt.tick_params(axis='both', labelsize=14, length=6, width=1.5)
            plt.ylim(Y_LIMIT)
            plt.legend(markerscale=1.2, fontsize=14)
            plt.show()

    def visual_analysis(self) -> list:
        plt.figure(figsize=WIDE_FIG_SIZE)
        # 1     2    3     4      5     6       7     8      9    10    11    12     13
        split_values = [0.25, 0.08, 0.27, 0.19, 0.275, 0.075, 0.265, 0.125, 0.34, 0.25, 0.26, 0.26, 0.18]
        plt.scatter(self.neural_interval[0], self.neural_metric[0], s=3)
        values = []

        split = np.min(self.neural_interval[0])
        i = 0
        for value in split_values:
            split += value
            values.append(split)
            if i < 1:
                plt.axvline(x=split, linestyle=f'--', color="orange", label="Segments")
            else:
                plt.axvline(x=split, linestyle=f'--', color="orange")
            i += 1
        #

        plt.xlabel("Time in seconds")

        plt.ylabel("Electrical current")
        plt.savefig("aligned_cycles.png", dpi=300, bbox_inches="tight")
        plt.legend(framealpha=1, fontsize=LABEL_FONT_SIZE)
        plt.show()

        return values

    def plot_aligned_cycle(self) -> None:
        conc_neural = np.concatenate(self.neural_metric)
        conc_time = []
        for n_t in self.neural_interval:
            conc_time.append(n_t - np.min(n_t))

        conc_time = np.concatenate(conc_time)
        plt.figure(figsize=WIDE_FIG_SIZE)
        plt.scatter(conc_time, conc_neural, s=0.06, alpha=0.45, label="Electrical current")

        plt.xlabel("Time in seconds", fontsize=LABEL_FONT_SIZE)
        plt.ylabel("Electrical current", fontsize=LABEL_FONT_SIZE)
        # plt.savefig("aligned_cycles.png", dpi=300, bbox_inches="tight")
        legend = plt.legend(framealpha=1, fontsize=16)
        plt.tick_params(axis='both', labelsize=14, length=6, width=1.5)
        for handle in legend.legendHandles:
            handle.set_alpha(1)
            handle.set_sizes([100])
        plt.show()

    def simple_scatter(self, t: np.ndarray, y: np.ndarray, label: str) -> None:
        plt.figure(figsize=WIDE_FIG_SIZE)
        plt.scatter(t - np.min(t), y, s=3, label=label)
        plt.xlabel("Time in seconds", fontsize=LABEL_FONT_SIZE)
        plt.ylabel("Electrical current", fontsize=LABEL_FONT_SIZE)
        plt.tick_params(axis='both', labelsize=14, length=6, width=1.5)
        plt.ylim(Y_LIMIT)
        legend = plt.legend(markerscale=1, fontsize=16)
        for handle in legend.legendHandles:
            handle.set_alpha(1)
            handle.set_sizes([100])
        plt.show()


class Analyze(ProcessCSV):

    """
    This class is responsible for returning the statistics of the image and CNN phase,
    It also displays scatter plots of the different phases.
    """

    def __init__(self):
        super().__init__()

    def get_image_stat(self, showPlot=False) -> (np.ndarray, np.ndarray):
        stat_image = []
        sample = None
        for image_t, image_y in zip(self.image_interval, self.image_metric):
            if showPlot:
                self.simple_scatter(image_t, image_y, label="Image phase Electrical Current")
            if len(image_y) > 37:
                stat_image.append(
                    [len(image_y), np.mean(image_y), np.std(image_y), np.min(image_y), np.max(image_y)])
            else:
                sample = image_t
                break

        return np.asarray(stat_image), sample

    def get_neural_stat(self, showPlot=False) -> np.ndarray:
        neural_stat = []
        for n_t, n_y in zip(self.neural_interval, self.neural_metric):
            if showPlot:
                self.simple_scatter(n_t - np.min(n_t), n_y, label="CNN phase Electrical Current")

                neural_stat.append([len(n_y), np.mean(n_y), np.std(n_y), np.min(n_y), np.max(n_y)])
        return np.asarray(neural_stat)
