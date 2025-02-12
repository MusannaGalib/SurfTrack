import os
import shutil
import glob
import time

# Default values for retries and delay
DEFAULT_RETRIES = 10  # Adjust this value to change the number of retries
DEFAULT_DELAY = 10    # Adjust this value to change the delay between retries in seconds

def delete_scripts_from_all_folders(base_directory, matlab_script_names, runner_script_name, retries=DEFAULT_RETRIES, delay=DEFAULT_DELAY):
    """
    Deletes specified scripts from all folders under the base directory if a certain condition is met.
    The condition is the existence of the 'Tracked_surface_compare_plots' folder, checked with retries.

    Parameters:
    - base_directory: The base directory where video folders are located.
    - matlab_script_names: List of MATLAB script filenames to delete.
    - runner_script_name: The name of the runner Python script to delete.
    - retries: Number of retries to check for the condition.
    - delay: Delay in seconds between retries.
    """
    print("Starting cleanup...")
    for video_folder in glob.glob(os.path.join(base_directory, '*/')):
        if os.path.isdir(video_folder):
            success = False
            attempts = 0
            while attempts < retries and not success:
                if 'Tracked_surface_compare_plots' in os.listdir(video_folder):
                    for script_name in matlab_script_names + [runner_script_name]:
                        script_path = os.path.join(video_folder, script_name)
                        if os.path.exists(script_path):
                            os.remove(script_path)
                            print(f"Deleted {script_name} from {video_folder}")
                    success = True
                else:
                    print(f"'Tracked_surface_compare_plots' folder not found in {video_folder}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    attempts += 1
            if not success:
                print(f"Failed to find 'Tracked_surface_compare_plots' folder in {video_folder} after {retries} attempts. Skipping cleanup for this folder.")

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    matlab_script_names = ['trracking_master_code.m', 'Tracked_surface_compare.m']
    runner_script_name = 'run.py'
    delete_scripts_from_all_folders(base_directory, matlab_script_names, runner_script_name, retries=DEFAULT_RETRIES, delay=DEFAULT_DELAY)
    print("Cleanup is complete.")
