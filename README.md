# Research Project
**Cycle-aware, segment wise statistical Industrial CPS emulator**

This is a cycle-aware, segment-wise pipeline for an emulator that synthezises electrical traces based on segment gathered statistics.

Briefly, the emulator takes the following steps:

1. Segment the traces based on domain specific knowledge
2. Segment-wise compute the residuals with respect to the median
3. Extract statistics that can describe the residual's non-Gaussian distribution observed per segment (also per cycle)
4. Synthezise residuals based on those statistics → interpolate a line based on the reconstructed quantiles (inverse CDF) and then randomly sample
4. Get a baseline trace by randomly picking a signal, then smooth it using the savgol filter
5. Add the synthezised residuals
6. Validate/Display the results andd/or persistently stream to a CSV file

---

## To use the emulator

1. Create a conda env, the emulator currently runs on python **3.10.9**
2. Install the packets in the `requirements.txt` file, the versions of numpy and pandas **MUST** be compatible
3. Run the `main.py` file normally

---

## Current problems with the emulator

Below there is a list explaining what are the current problems/limitations that the emulator is facing.

### Residual synthesis / statistics collection

- The statistics gathered output a **"block-like"** texture, this is the result of the quantile interpolation  
  that is computed as a diagonal line

- To extract/create baseline signal, all traces are fitted on a fixed size  
  (max. observed trace size) grid. This method cannot output different length traces.

- The baseline trace is computed directly from a collected one, there is no synthesis of a new base-trace based
on the statistics observed of the previous trace.

- There is no temporal-preservation,both cycle-wise and segment-wise, taken into account
when a trace is synthesised.

---

### Overall program

- There is no user interface
- There is another method of creating a surrogate, however it has not been documented or validated
- The program uses a multi threaded TUI, which creates a new thread based on the task chosen. Currently it's only 
validation and persistent CSV writing. When the program is stopped, the threads are forced to stop using os.exit, which
is not thread safe. This will not save partial validation/CSV output.
---

### Output

- No persistent output to a platform like Apache Kafka  
