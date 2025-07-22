from skimage import io
import animatplot as amp
import matplotlib.pyplot as plt

video_ds = io.imread('C:\\Users\\zaggila\\Documents\\pixelNMF\\data\\sz92_2024-06-06_a_cell_control\\AVG_cam_crop_5x_avg.tif') # initialise tiff

block = amp.blocks.Imshow(video_ds)
anim = amp.Animation([block])

anim.controls()
# anim.save_gif('ising')
plt.show()


fig, axs = plt.subplots(2,2, figsize=(15,10), dpi=300, sharex=True) #and remove down of first 

axs[0].imshow(sorted_data_by_mod_idx_active_rest_state_df_f[::-1], vmin=0, vmax=np.percentile(sorted_data_by_mod_idx_active_rest_state_df_f,95), cmap='binary', aspect='auto' )
axs[0].set_ylabel('ROIs', fontsize=35, labelpad=10)
axs[0].set_title('mod_idx_active_rest_rois (df/f)', fontsize=35)
axs[0].set_xticks(ticks=frame_ticks, labels=second_ticks, fontsize=12)

#threshold and binarise

trio_motion_energy = np.where(gaussian_filter1d(motion_energy, sigma=10) < threshold_motion_energy, 0, 1)
idx_active_state_motion_energy= np.where(trio_motion_energy ==1)
idx_inactive_state_motion_energy = np.where(trio_motion_energy ==0)
trio_motion_energy[inds_twitches] = -1

axs[1].plot(trio_motion_energy, color='darkorange', linewidth=1)
axs[1].plot(motion_energy, c='cyan', linewidth=1) # was computed on low_motion_energy (no smoothing)
axs[1].set_title('twitch detection on motion energy', fontsize=30)
axs[1].axhline(y=threshold_twicthes, color='red', linestyle='--', label='twitch detection threshold')
axs[1].set_xticks(ticks=frame_ticks, labels=second_ticks, fontsize=12)
axs[1].set_xlabel('Time(s)', fontsize=30)
axs[1].legend()

plt.subplots_adjust(hspace=0.4)

plt.savefig(save_dir_videography + 'mod_idx_active_rest_sorted_raster_and_twitch_detection_DF_F.png')
plt.show()