import os
import h5py
import efel
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Config
# input_folder = "/pscratch/sd/s/sdough/developing_neuron_simraw_H5_step8"
# output_file = "filtered_voltages_simRaw_step8.h5"

input_folder = "/pscratch/sd/s/sdough/optimizing_all_params"
output_file = "filtered_voltages_simRaw_optimizing_all.h5"

# input_folder = "/pscratch/sd/s/sdough/Neuron_Latest_Pipeline/DL4neurons2/simRaw"
# output_file = "filtered_voltages_simRaw.h5"

# eFEL settings
spike_threshold = 10
efel.setDoubleSetting("interp_step", 0.1)
efel.setThreshold(10)
MIN_AP_AMPLITUDE = 60.0
MAX_SPIKES = 50
MIN_AP_WIDTH = 0.5
MAX_AP_WIDTH = 4.0

# Open output file
with h5py.File(output_file, "w") as out_f:
    filtered_group = out_f.create_group("filtered_data")
    spike_counts = []

    for filename in tqdm(os.listdir(input_folder)):
        if not filename.endswith(".h5"):
            continue

        filepath = os.path.join(input_folder, filename)
        with h5py.File(filepath, "r") as f:
            meta = json.loads(f['meta.JSON'][0].decode())
            dt = meta['timeAxis']['step']
            time = np.arange(0, meta['num_time_bins'] * dt, dt)
            probe_names = meta['simu_info']['full_prob_names']

            voltages = f['volts'][:]
            phys_par = f['phys_par'][:]

            for trace_idx in range(voltages.shape[0]):
                for probe_idx, probe_name in enumerate(probe_names):
                    voltage_trace = voltages[trace_idx, :, probe_idx, 0]

                    if np.isnan(voltage_trace).any():
                        continue

                    trace = {
                        'T': time.tolist(),
                        'V': voltage_trace.tolist(),
                        'stim_start': [0],
                        'stim_end': [time[-1]],
                    }

                    features = efel.getFeatureValues(
                        [trace],
                        ['Spikecount', 'AP_amplitude', 'AP_width']
                    )[0]

                    spike_count = 0
                    valid_amplitude = False
                    valid_width = False
                    max_amp = float('-inf')
                    mean_width = None

                    if features is not None:
                        spike_count = features.get('Spikecount', [0])[0]

                        # Check amplitude validity
                        ap_amps = features.get('AP_amplitude', [])
                        if ap_amps is not None and len(ap_amps) > 0:
                            max_amp = max(ap_amps)
                            valid_amplitude = max_amp >= MIN_AP_AMPLITUDE

                        # Check AP width validity
                        ap_widths = features.get('AP_width', [])
                        if ap_widths is not None and len(ap_widths) > 0:
                            mean_width = np.nanmean(ap_widths)
                            valid_width = MIN_AP_WIDTH <= mean_width <= MAX_AP_WIDTH

                    spike_counts.append(spike_count)

                    # Filtering condition
                    if spike_threshold <= spike_count <= MAX_SPIKES and valid_amplitude and valid_width:
                        print(f"{filename} trace {trace_idx} probe {probe_name} "
                                f"spikes={spike_count}, max_amp={max_amp:.2f} mV, "
                                f"mean_width={mean_width:.2f} ms")

                        trace_name = f"{filename}_trace{trace_idx}_{probe_name}"
                        grp = filtered_group.create_group(trace_name)
                        grp.create_dataset("voltage", data=voltage_trace)
                        grp.create_dataset("time", data=time)
                        grp.create_dataset("phys_par", data=phys_par[trace_idx])
                        grp.attrs["spike_count"] = spike_count
                        grp.attrs["max_ap_amplitude"] = max_amp
                        grp.attrs["mean_ap_width"] = mean_width

    # Save spike count histogram
    plt.hist(spike_counts, bins=20)
    plt.title("Spike Count Distribution")
    hist_path = output_file.replace('.h5', '_spike_histogram.png')
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved spike histogram to: {hist_path}")
    
# Export baselines to CSV
with h5py.File(output_file, 'r') as f:
    baseline_data = []
    for trace in f['filtered_data']:
        baseline_data.append({
            'params': f['filtered_data'][trace]['phys_par'][()],
            'spikes': f['filtered_data'][trace].attrs.get('spike_count', 0),
            'source': trace
        })

    pool = baseline_data
    pool_sorted = sorted(pool, key=lambda x: x['spikes'])
    n_select = min(10, len(pool_sorted))
    indices = np.linspace(0, len(pool_sorted) - 1, n_select, dtype=int)
    selected_baselines = [pool_sorted[i] for i in indices]

    baseline_df = pd.DataFrame(
        [b['params'] for b in selected_baselines],
        columns=meta['parName']
    )
    baseline_df['source_trace'] = [b['source'] for b in selected_baselines]
    baseline_df['spike_count'] = [b['spikes'] for b in selected_baselines]

    baseline_csv = output_file.replace('.h5', '_baselines.csv')
    baseline_df.to_csv(baseline_csv, index=False)
    print(f"Saved 10 baselines to: {baseline_csv}")

    out_dir = output_file.replace('.h5', '_NewBase_CSVs')
    os.makedirs(out_dir, exist_ok=True)

    for i, b in enumerate(selected_baselines):
        vals = b['params']
        df_vals = pd.DataFrame({
            'Parameters': meta['parName'],
            'Values': vals
        })
        csv_path = os.path.join(out_dir, f"NewBase_{i:02d}.csv")
        df_vals.to_csv(csv_path, index=False)
        print(f"Wrote baseline {i} to {csv_path}")