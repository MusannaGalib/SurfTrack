import subprocess
import os
import shutil
import glob
import time


def report_processing_completion(base_directory, matlab_script_names, runner_script_name):
    total_videos_processed = 0

    # Iterate over each .mp4 file in the base directory to count them
    for video_file in glob.glob(os.path.join(base_directory, '*.mp4')):
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f"Video processed: {video_name}")
        total_videos_processed += 1

    print(f"Total videos processed: {total_videos_processed}")
    
    # Additional information or checks can be added here
    
    # Adding a delay to ensure all processes complete before deletion
    print("Pausing before cleanup to ensure all processes have completed...")
    time.sleep(0)  # Adjust the sleep time as necessary


def delete_scripts_from_all_folders(base_directory, matlab_script_names, runner_script_name):
    print("Starting cleanup...")
    for video_folder in glob.glob(os.path.join(base_directory, '*/')):
        if os.path.isdir(video_folder):
            for script_name in matlab_script_names + [runner_script_name]:
                script_path = os.path.join(video_folder, script_name)
                if os.path.exists(script_path):
                    os.remove(script_path)
                    print(f"Deleted {script_name} from {video_folder}")

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
    # Cleanup after all videos have been processed

    # Call the reporting function before deletion
    report_processing_completion(os.path.dirname(os.path.abspath(__file__)), matlab_script_names, runner_script_name)
    
    # Proceed with deletion after the reporting and pause
    delete_scripts_from_all_folders(os.path.dirname(os.path.abspath(__file__)), matlab_script_names, runner_script_name)
    print("All videos have been processed and cleanup is complete.")