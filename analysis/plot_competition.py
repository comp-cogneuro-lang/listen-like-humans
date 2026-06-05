import pandas as pd
import os
import argparse
from matplotlib import pyplot as plt
import numpy as np

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    print(df.head())
    return df

def preprocess_rhyme(rhyme_data):
    rhyme_series = pd.Series(rhyme_data)
    # Remove the unusual values at the start and the end
    if len(rhyme_series) > 5:
        rhyme_series.iloc[:5] = rhyme_series.iloc[5]
    rhyme_series.iloc[-20:] = rhyme_series.iloc[-30:-20].mean()
    # do the smoothing 
    rhyme_series = rhyme_series.rolling(window=10, center=True, min_periods=1).mean()
    return rhyme_series.tolist()

def process_mean_activations(df, max_timestep, smooth_target=False):
    df_mean = df.groupby('Category').mean(numeric_only=True)
    df_mean_dict = df_mean.to_dict(orient='index')
    category2activation = {}
    
    for category, data in df_mean_dict.items():
        print(f"Category: {category}")
        if category not in category2activation:
            category2activation[category] = []
        
        for idx, value in data.items():
            if idx == "Epoch": continue
            category2activation[category].append(value)
        category2activation[category] = category2activation[category][:max_timestep]
    
    if 'Rhyme' in category2activation:
        category2activation['Rhyme'] = preprocess_rhyme(category2activation['Rhyme'])
    
    if smooth_target and 'Target' in category2activation:
        target_series = pd.Series(category2activation['Target'])
        target_series = target_series.rolling(window=10, center=True, min_periods=1).mean()
        category2activation['Target'] = target_series.tolist()
        
    return category2activation

def plot_competition(category2activation, output_dir, max_mstime, frame2ms, max_timestep, fig_name='figure_0_competition.png'):
    print(f"Plotting competition to {output_dir}")
    plt.figure()
    for category, activations in category2activation.items():
        # plot the activation over time with x axis as time in ms
        time_axis = [i * frame2ms for i in range(len(activations[:max_timestep]))]
        plt.plot(time_axis, activations[:max_timestep], label=category)
        
        # make x axis lable more dense
        plt.xticks(range(0, max_timestep * frame2ms, 10 * frame2ms))
        
        # limit the x axis to max_mstime
        plt.xlim(0, max_mstime)
        
        print(f"len activations {category}: ", len(activations))

    plt.legend()
    # Exchange the left and right axis position
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    output_path = os.path.join(output_dir, fig_name)
    plt.savefig(output_path)
    print(f"Saved competition plot to {output_path}")
    plt.close()

def save_mean_csv(category2activation, output_dir, frame2ms):
    df_mean_fname = os.path.join(output_dir, 'competition_mean.csv')
    
    # Convert dict to df, ensuring consistent length requires care if lists differ
    # Assuming same length due to truncation in process_mean_activations
    df_mean = pd.DataFrame(category2activation)
    df_mean = df_mean.round(4)
    
    # Filter columns if they exist
    cols_to_keep = ['Target', 'Cohort', 'Rhyme', 'Anadrome']
    existing_cols = [c for c in cols_to_keep if c in df_mean.columns]
    # If we have extra columns, keep only the specific ones if present
    if existing_cols:
        df_mean = df_mean[existing_cols]
    
    df_mean.insert(0, 'Time', [i for i in range(len(df_mean))])
    df_mean.to_csv(df_mean_fname, index=False)
    print(f"Saved mean csv to {df_mean_fname}")

def plot_target_variability(df, output_dir, max_timestep, frame2ms):
    if 'Category' not in df.columns:
        print("Category column missing, skipping variability plot.")
        return
        
    activations_df = df[df['Category'] == 'Target']
    if activations_df.empty:
        print("No Target category found for variability plot.")
        return

    # Drop non-numeric columns
    cols_to_drop = ['Word', 'Speaker', 'Category']
    # Select only columns actually present
    present_drop_cols = [c for c in cols_to_drop if c in activations_df.columns]
    
    activations = activations_df.drop(columns=present_drop_cols).values.T  # shape (n_samples, T)

    # Compute median and percentiles
    # activations shape is (Time, Samples) after transpose? 
    # Original: activations = activations.drop(...).values.T
    # If original df has Shape (Samples, TimeColumns...), dropping gives (Samples, TimeColumns). 
    # .T gives (TimeColumns, Samples).
    
    timesteps_avail = min(activations.shape[0], max_timestep)
    
    # slicing to max_timestep
    activations_trunc = activations[:timesteps_avail, :]

    median = np.median(activations_trunc, axis=1)
    lower = np.percentile(activations_trunc, 5, axis=1)
    upper = np.percentile(activations_trunc, 95, axis=1)

    # Plot
    plt.figure(figsize=(10, 6))
    T_axis = [i * frame2ms for i in range(timesteps_avail)]
    
    plt.fill_between(T_axis, lower, upper, alpha=0.3, label='5th–95th percentile')
    plt.plot(T_axis, median, 'b-', label='Median')
    
    # Optional: plot a few individual traces with low opacity
    # activations_trunc is (Time, Samples)
    num_traces = min(20, activations_trunc.shape[1])
    for i in range(num_traces):
        plt.plot(T_axis, activations_trunc[:, i], 'k-', alpha=0.1)
        
    plt.xlabel('Time step')
    plt.ylabel('Probability')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Probability time series with variability')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'Target_probability_with_variability.png')
    plt.savefig(output_path)
    print(f"Saved variability plot to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot competition analysis.")
    parser.add_argument('input_path', nargs='?', 
                        default="experiments/nemotron_realtime/competition.csv",
                        help="Path to the competition CSV file.")
    parser.add_argument('--max_mstime', type=int, default=1000, help="Max time in ms.")
    parser.add_argument('--frame2ms', type=int, default=10, help="Milliseconds per frame.")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save outputs. (Defaults to input file directory.)")
    parser.add_argument('--smooth', action='store_true', help='Whether to smooth the target curve')
    args = parser.parse_args()

    input_path = args.input_path
    if args.output_dir:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(input_path)

    # Validate input path
    if not os.path.exists(input_path):
        print(f"Error: Input file found at {input_path}")
        return

    max_timestep = args.max_mstime // args.frame2ms
    
    print(f"Processing {input_path}...")
    try:
        df = load_data(input_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        raise ValueError("No competition data found.")

    print(f"Number of samples: {len(df)}")

    # Process and Plot Mean Competition
    category2activation = process_mean_activations(df, max_timestep, args.smooth)
    plot_competition(category2activation, output_dir, args.max_mstime, args.frame2ms, max_timestep)
    save_mean_csv(category2activation, output_dir, args.frame2ms)

    # Plot Variability
    plot_target_variability(df, output_dir, max_timestep, args.frame2ms)

if __name__ == "__main__":
    main()
