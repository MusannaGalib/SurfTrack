import os
import shutil
import glob

def delete_scripts_from_all_folders(base_directory, matlab_script_names, runner_script_name):
    print("Starting cleanup...")
    for video_folder in glob.glob(os.path.join(base_directory, '*/')):
        if os.path.isdir(video_folder):
            for script_name in matlab_script_names + [runner_script_name]:
                script_path = os.path.join(video_folder, script_name)
                if os.path.exists(script_path):
                    os.remove(script_path)
                    print(f"Deleted {script_name} from {video_folder}")

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    matlab_script_names = ['trracking_master_code.m', 'Tracked_surface_compare.m']
    runner_script_name = 'run.py'
    delete_scripts_from_all_folders(base_directory, matlab_script_names, runner_script_name)
    print("Cleanup is complete.")
