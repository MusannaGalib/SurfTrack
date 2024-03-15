import subprocess
import os
import shutil
import glob
import time


def setup_and_run_video_processing(matlab_script_names, runner_script_name):
    base_directory = os.path.dirname(os.path.abspath(__file__))

    # Iterate over each .mp4 file in the base directory
    for video_file in glob.glob(os.path.join(base_directory, '*.mp4')):
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        video_folder = os.path.join(base_directory, video_name)

        # Create a folder for the video if it doesn't already exist
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        # Copy the .mp4 file to the newly created folder
        dest_video_path = os.path.join(video_folder, os.path.basename(video_file))
        shutil.copy(video_file, dest_video_path)

        # Copy MATLAB scripts and the Python runner script to the video folder
        for script_name in matlab_script_names + [runner_script_name]:
            src_script_path = os.path.join(base_directory, script_name)
            dest_script_path = os.path.join(video_folder, script_name)
            shutil.copy(src_script_path, dest_script_path)

        # Execute the runner Python script within the video folder context
        current_directory = os.getcwd()  # Remember the current directory
        os.chdir(video_folder)  # Change to the video's folder
        try:
            subprocess.check_call(['python', runner_script_name])
        except subprocess.CalledProcessError as e:
            print(f"Error executing {runner_script_name} for video {video_name}: {e}")
        finally:
            os.chdir(current_directory)  # Always revert back to the original directory



if __name__ == "__main__":
    matlab_script_names = ['trracking_master_code.m', 'Tracked_surface_compare.m']
    runner_script_name = 'run.py'  # Assuming this is the name of your current Python script
    print("Starting video processing...")
    setup_and_run_video_processing(matlab_script_names, runner_script_name)
    print("All videos have been processed and cleanup is complete.")