import csv
import os
import numpy as np
from process_csv import ProcessCSV
from time import sleep

VOLTAGE = 5.0

DT = 0.00025
HEADERS = ["Timestamp(S)", "Arc Main Current(A)", "Arc Main Voltage(V)", "Arc Main Voltage(J)", "Arc UART(TXT)"]


class PersistentCSVStreamer(ProcessCSV):
    """
    Persistent streamer that appends samples to a CSV file with headers:
    Timestamp, Current, Voltage, Energy, TXT
    """

    def __init__(self, filepath: str, syn_traces: np.ndarray, t_surr: np.ndarray, stop_event):

        super().__init__()
        self.syn_traces = syn_traces
        self.t_syn = t_surr
        self.out_path = os.path.join(self._OUTPUT_FILE, fr"{filepath}.csv")
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        self.stop_event = stop_event

    def stream_data(self):
        while not self.stop_event.is_set():
            energy_j = 0.0
            idx = 0

            with open(self.out_path, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow(HEADERS)

                for cycle in zip(self.syn_traces):
                    if idx > 0:
                        time = self.t_syn + np.max(self.t_syn)
                    else:
                        time = self.t_syn
                    time = np.round(time, 5)
                    for c, t in zip(cycle[0], time):
                        sleep(DT)
                        energy_j += VOLTAGE * c * DT

                        writer.writerow([
                            f"{t:.5f}",
                            f"{c:.5f}",
                            f"{VOLTAGE:.5f}",
                            f"{energy_j:.5f}",
                            None
                        ])

                        idx += 1
            pass

