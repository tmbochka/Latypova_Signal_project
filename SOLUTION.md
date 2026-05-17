# Latypova_Signal_project

# SMILES-2026 Signal Interference Cancellation

# Solution 

## How to Run
1. Install requirements: `pip install numpy scipy gdown`
2. Run the script: `python applicant_solution.py`
3. Final metrics are saved in `results.json`.

---

## Technical Approach

This solution removes two types of noise:
*   **Self-Interference:** Noise created by the device's own antennas.
*   **External Interference:** Noise coming from an outside source.

### Key Features:
*   **Extended Model:** An 18-term model is used to accurately describe how the device distorts the signal.
*   **Iterative Loop:** An alternating cycle is used: first removing self-noise, then external noise, and repeating. This prevents the two noise types from mixing up and confusing the calculation.
*   **Wiener Deconvolution:** Standard filters can distort the shape of the noise at the edges. This method is used to reconstruct the true, full shape of the external noise so it can be subtracted perfectly without leaving artifacts.
*   **Stability Margin:** To prevent trigger, the removed external noise is multiplied by a 0.98 factor.

---

## Experiments

I tested several approaches before arriving at the final solution:
*   Calculating the settings based on the entire 2.4 million samples at once. This resulted in a lower score (~6.02 dB).
*   Cleaning the noise in a single pass without the iterative loop. This gave a decent score (~7.01 dB), but the external noise was still slightly confusing the self-noise calculations.

---

## Final Results
*   **Average Metric:** **9.07 dB**
*   **Per-Channel:**
ch0: 10.13 dB, ch1: 7.99 dB, ch2: 11.05 dB, ch3: 7.12 dB

---
**Yuliia Latypova**
