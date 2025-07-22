import matplotlib.pyplot as plt
import seaborn as sns


def plot_binary_motion_energy_states(smoothed_motion_energy, 
                                     naive_binary, 
                                     cleaned_binary, 
                                     threshold_motion_energy,
                                     frame_rate=30, 
                                     save_path=None):
    """
    Plots the smoothed motion energy signal with thresholds and both binary states:
    naive (before gap-filling) and cleaned (after gap-filling).

    Parameters:
    - smoothed_motion_energy: np.array, motion energy signal
    - naive_binary: np.array, binary signal before gap filling
    - cleaned_binary: np.array, binary signal after gap filling
    - threshold_motion_energy: float, threshold used for binarisation
    - frame_rate: int, how many frames per second (for x-axis ticks)
    - save_path: optional path to save the plot
    """
    seconds = np.arange(len(smoothed_motion_energy)) / frame_rate

    plt.figure(figsize=(10, 3), dpi=300)

    plt.plot(seconds, smoothed_motion_energy, label='mot_en', linewidth=1.5)
    plt.axhline(threshold_motion_energy, color='red', linestyle='--', label='Threshold')

    plt.plot(seconds, naive_binary * smoothed_motion_energy.max(), 
             label='Binary', color='orange', linewidth=1, alpha=0.7)

    plt.plot(seconds, cleaned_binary * smoothed_motion_energy.max(), 
             label='Binary (gap-filled)', color='green', linewidth=2, alpha=0.7)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Motion Energy / Binary State', fontsize=12)
    plt.title('Motion Energy Binarisation Comparison', fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_active_motion_classification_subplots(
    motion_signal,
    smoothed_motion_signal,
    bin_motion_energy,
    bin_long_active_motion,
    bin_short_active_motion,
    long_active_motion_inds,
    short_active_motion_inds,
    threshold,
    save_path=None,
):
    """
    Plot classified motion energy traces with subplots for visual inspection.

    Parameters:
    - motion_signal: Raw motion energy.
    - smoothed_motion_signal: Smoothed motion energy.
    - bin_motion_energy: Initial binary signal of active vs. inactive motion (before filtering).
    - bin_long_active_motion: Binary trace of long active motions only.
    - bin_short_active_motion: Binary trace of short active motions (1–3s).
    - long_active_motion_inds: Indices of long motions.
    - short_active_motion_inds: Indices of short motions.
    - threshold: Motion energy threshold used for binarization.
    - save_path: Optional path to save the figure (e.g., './figs/plot.png').
    - title_prefix: Optional prefix to prepend to each plot title.
    """

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), dpi=100)

    # plot motion energy with markers for short/long motions 
    axes[0].plot(smoothed_motion_signal, label="mot_en", linewidth=2)
    axes[0].axhline(y=threshold, color='red', linestyle='--', label='threshold', linewidth=2)

    if len(long_active_motion_inds) > 0:
        axes[0].plot(long_active_motion_inds, smoothed_motion_signal[long_active_motion_inds],
                      color='orange', label="long motions > 3s")

    if len(short_active_motion_inds) > 0:
        axes[0].plot(short_active_motion_inds, smoothed_motion_signal[short_active_motion_inds],
                     color='red', label="short motions 1-3s")

    axes[0].set_ylabel('mot_en', fontsize=15, labelpad=10)
    axes[0].set_xlabel('time (frames)', fontsize=15, labelpad=10)
    axes[0].legend()
    axes[0].set_title(f'active motions - long vs short', fontsize=18)

    # binary classification all active motions
    axes[1].plot(smoothed_motion_signal, label="mot_en", alpha=0.6)
    axes[1].plot(bin_motion_energy, color='orange', linewidth=2, label="short + long active motions")
    axes[1].axhline(y=threshold, color='red', linestyle='--', label='threshold', linewidth=1)
    axes[1].set_ylabel('mot_en', fontsize=15, labelpad=10)
    axes[1].set_xlabel('time (frames)', fontsize=12, labelpad=10)
    axes[1].set_title(f'bin active motions (long+short)', fontsize=18)

    # binary classification only long motions plotted 
    axes[2].plot(smoothed_motion_signal, label="mot_en", alpha=0.6)
    axes[2].plot(bin_long_active_motion, color='orange', linewidth=2, label="long active motions")
    axes[2].axhline(y=threshold, color='red', linestyle='--', label='threshold', linewidth=1)
    axes[2].set_ylabel('mot_en', fontsize=15, labelpad=10)
    axes[2].set_xlabel('time (frames)', fontsize=15, labelpad=10)
    axes[2].set_title(f'bin active motions (long)', fontsize=18)

    for ax in axes:
        ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")

    plt.show()


def plot_detected_twitches(
    motion_energy,
    smoothed_motion_energy,
    bin_motion_energy,
    threshold_twitches,
    threshold_motion_energy,
    inds_twitches,
    frame_ticks,
    second_ticks,
    save_dir_videography=None
):
    """
    Plot the motion energy and smoothed motion energy with twitch detection.

    Parameters:
    - motion_energy: Raw motion energy signal.
    - smoothed_motion_energy: Smoothed motion energy signal.
    - threshold_twitches: Threshold for twitch detection.
    - threshold_motion_energy: Threshold for binary motion energy.
    - inds_twitches: Indices where twitch segments start (onsets).
    - frame_ticks: Frame indices for x-axis.
    - second_ticks: Time ticks in seconds for x-axis.
    - save_dir_videography: Optional directory to save the figure.
    """
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 7), dpi=300)

    # Plot raw and smoothed motion energy
    axs[0].plot(motion_energy, color='orange', linewidth=2, label='mot_en')
    axs[0].plot(smoothed_motion_energy, color='blue', linewidth=1.5, label='mot_en smoothed')
    axs[0].axhline(y=threshold_twitches, color='red', linestyle='--', label='Twitch threshold')
    axs[0].set_xticks(ticks=frame_ticks)
    axs[0].set_xticklabels(second_ticks, fontsize=12)
    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].set_ylabel('mot_en', fontsize=15)
    axs[0].set_title('mot_en - twitch detection', fontsize=18)

    trio_motion_energy = bin_motion_energy.copy()
    trio_motion_energy[inds_twitches] = -1  # mark twitch segments with -1

    # Plot raw motion energy and binary motion energy (including twitch detection)
    axs[1].plot(motion_energy, color='blue', linewidth=1, label='mot_en')
    axs[1].plot(trio_motion_energy, color='darkorange', linewidth=1, label='3 states on mot_en')
    axs[1].axhline(y=threshold_twitches, color='red', linestyle='--', label='twitch threshold')
    axs[1].set_xticks(ticks=frame_ticks)
    axs[1].set_xticklabels(second_ticks, fontsize=12)
    axs[1].set_xlabel('Time (s)', fontsize=15)
    axs[1].set_title('Twitch Detection', fontsize=18)
    axs[1].legend(loc='upper right', fontsize=10)

    # Adjust layout and save the plot
    plt.subplots_adjust(hspace=0.6)

    if save_dir_videography:
        plt.savefig(save_dir_videography + 'trio_motion_energy_including_twitches.png')
        print(f"Plot saved to: {save_dir_videography + 'trio_motion_energy_including_twitches.png'}")

    plt.show()
    

def plot_motion_trace(motion, smoothed_motion=None, title='Motion trace', save_path=None):
    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(motion, label='Raw motion', alpha=0.5)
    if smoothed_motion is not None:
        plt.plot(smoothed_motion, label='Smoothed motion', linewidth=2)
    plt.title(title)
    plt.xlabel('Frames')
    plt.ylabel('Motion energy')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_twitch_locations(motion_trace, twitch_indices, title='Detected twitches', save_path=None):
    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(motion_trace, label='Motion energy', alpha=0.5)
    plt.scatter(twitch_indices, motion_trace[twitch_indices], color='red', label='Twitches', s=10)
    plt.title(title)
    plt.xlabel('Frames')
    plt.ylabel('Motion energy')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
