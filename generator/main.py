import threading
import traceback

from emulator import Emulator
from model import *
from output_handler import PersistentCSVStreamer
from process_csv import ProcessCSV, Analyze
from validate import Validate
from concurrent.futures import ThreadPoolExecutor
import os


def break_trace_time(y: np.ndarray, t: np.ndarray, values: np.ndarray, res: bool = False) -> (np.ndarray, np.ndarray):
    """
    This function segments the trace based on the domain specific knowledge observed timestamps
    :param y: The array that contains the values
    :param t: The array that contains time values
    :param values: The time-wise split values
    :param res: Boolean for residual calculation
    :return:
    """
    slice_idx = np.where(np.isin(t, np.round(values, 3)))[0]
    sliced_y = np.array_split(y, slice_idx)
    sliced_t = np.array_split(t, slice_idx)
    slice_stat = []
    for s, t in zip(sliced_y, sliced_t):
        if res:
            slice_stat.append([*extract_slice_features(s - np.median(s), t, show=False)])
        else:
            slice_stat.append([*extract_slice_features(s, t=None, show=False)])

    return np.vstack(slice_stat), slice_idx


def break_trace_index(y: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Same function as the one above, however this one uses segment indexes instead of time values
    slice_idx = np.where(np.isin(np.flatnonzero(y), np.round(values, 3)))[0]
    sliced_y = np.array_split(y, slice_idx)
    slice_stat = []
    for s in sliced_y:
        slice_stat.append([*extract_slice_features(s, t=None, show=False)])

    return np.vstack(slice_stat)


def make_cube(y: np.ndarray, segment_idx: np.ndarray) -> np.ndarray:
    """
    The function makes a cube based on the domain specific knowledge observed indexes, as well as the actual data array.
    :param y: Array with electric current values
    :param segment_idx: Indexes of the segment

    :return: a 3D numpy matrix.
            Rows -> cycles
            Columns -> statistics
            Depth -> Segments
    """
    syn_block = []
    for surr in y:
        syn_slice = break_trace_index(surr, segment_idx)
        syn_3d = syn_slice.reshape(syn_slice.shape[0], 1, syn_slice.shape[1])
        syn_block.append(syn_3d)

    return np.concatenate(syn_block, axis=1)


def run_job(name, fn, *args, **kwargs):
    if name == "TASK-3":
        print(f"\n[{name} START] Persistent output -> validation.csv")
    try:
        fn(*args, **kwargs)
        if name == "TASK-1":
            print(f"\n[{name} DONE] Validation performed, check: validation.csv")
        elif name == "TASK-3":
            print(f"\n[{name} DONE] Persistent output done, check: output_traces.csv")
    except Exception:
        print(f"\n[TASK FAIL] {name}")
        traceback.print_exc()

def tui(n_m: np.ndarray, n_t: np.ndarray, s_trace: np.ndarray, n_cube: np.ndarray, s_cube: np.ndarray):
    print("Press q to quit. Choose one option: ")

    time_syn = emulator.create_timestamp(0, 12345)
    stop_event = threading.Event()
    validate = Validate(stop_event)
    executor = ThreadPoolExecutor(max_workers=2)
    tasks = {}
    task_count = 0

    while True:
        print("1.Perform all validation")
        print("2.Display synthetic traces ")
        print("3.Stream synthetic traces to CSV")
        user_inp = input("Your choice: ").strip()
        if user_inp == "1":
            task_count +=1
            jid = f"TASK-{task_count}"
            future = executor.submit(
                run_job, jid,
                validate.perform_all_validation,
                s_trace, n_m, s_cube, n_cube
            )
            tasks[jid] = future
            print(f"Started validation process: {jid}")

        elif user_inp == "2":
            nr = int(input("How many results to display: "))
            validate.display_results(time_syn, s_trace, n_m, n_t, nr)


        elif user_inp == "3":
            task_count +=1
            jid = f"TASK-{task_count}"
            streamer = PersistentCSVStreamer("output_traces", s_trace, time_syn, stop_event)
            future = executor.submit(run_job, jid, streamer.stream_data)
            tasks[jid] = future

        elif user_inp == "q":
            print("\nShutting down...")
            stop_event.set()
            executor.shutdown(wait=False, cancel_futures=True)
            break
        else:
            print("Invalid character")

    print("Thank you for using the emulator, goodbye")
    #not safe
    os._exit(0)


if __name__ == "__main__":

    """
    This is the main loop where the program run. 
    In short, this loop is responsible for
    1. Collect metrics
    2. Extract residual features
    3. Synthesise residuals
    4. Validate/display or stream persistently
    """

    # Main branch
    process = ProcessCSV()
    analyzer = Analyze()
    stat_image = analyzer.get_image_stat()
    stat_neural = analyzer.get_neural_stat()

    neural_metric, neural_interval = process.neural_metric, process.neural_interval

    cycle_block = []
    split_values = [0.25, 0.08, 0.27, 0.19, 0.275, 0.075, 0.265, 0.125, 0.34, 0.25, 0.26, 0.26, 0.18]
    res_block = []
    segment_idx = []

    for n_t, n_y in zip(neural_interval, neural_metric):
        # get neural info
        split_t = np.min(n_t) + np.cumsum(split_values)
        n_sliced, slice_idx = break_trace_time(n_y, n_t, split_t)
        slices_3d = n_sliced.reshape(n_sliced.shape[0], 1, n_sliced.shape[1])
        segment_idx.append(slice_idx)
        cycle_block.append(slices_3d)
        assert len(slice_idx) == 13

        # get residual info
        res_sliced, _ = break_trace_time(n_y, n_t, split_t, res=True)
        res_3d = res_sliced.reshape(res_sliced.shape[0], 1, res_sliced.shape[1])
        res_block.append(res_3d)

    neural_cube = np.concatenate(cycle_block, axis=1)
    res_cube = np.concatenate(res_block, axis=1)
    emulator = Emulator()

    syn_trace = emulator.synthesize_surrogates(res_cube, segment_idx[0], 30)
    # sh_surrogate, surr_cycle = emulator.shuffle_surrogate(12345, 31, 10)

    syn_cube = make_cube(syn_trace, segment_idx[0])
    tui(neural_metric, neural_interval, syn_trace, neural_cube, syn_cube)
