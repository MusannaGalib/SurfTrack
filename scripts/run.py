import subprocess
import os
import time

def run_matlab_script(script_name, npics):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    try:
        if os.path.isfile(script_path):
            # Write npics to a text file
            with open('variables.txt', 'w') as f:
                f.write(str(npics))

            # Replace 'matlab' with the path to your MATLAB executable if it's not in the system PATH
            process = subprocess.Popen(['C:/Program Files/MATLAB/R2023b/bin/matlab', '-nosplash', '-nodesktop', '-r', f"run('{script_path}');exit;"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if stderr:
                print(stderr.decode("utf-8"))
        else:
            print(f"Error: MATLAB script '{script_name}' not found in the same folder as the Python script.")
    except FileNotFoundError:
        print("MATLAB executable not found. Make sure MATLAB is installed and added to the system PATH.")

# Function to create a folder if it doesn't exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to check if TrackedData.mat exists
def tracked_data_exists():
    return os.path.isfile(os.path.join(os.path.dirname(__file__), 'TrackedData.mat'))

# Example usage:
if __name__ == "__main__":
    # Change the current working directory to the directory of the Python script
    os.chdir(os.path.dirname(__file__))

    matlab_script_name_1 = 'trracking_master_code.m'
    matlab_script_name_2 = 'Tracked_surface_compare.m'

    # Define how many images you want from the video
    npics = 10

    run_matlab_script(matlab_script_name_1, npics)  # Run the first MATLAB script

    # Wait until TrackedData.mat is generated
    print("Waiting for TrackedData.mat to be generated...")
    while not tracked_data_exists():
        time.sleep(1)  # Check every second

    print("TrackedData.mat exists.")

    # Check if the second MATLAB script exists
    if os.path.isfile(os.path.join(os.path.dirname(__file__), matlab_script_name_2)):
        print("Second MATLAB script exists.")
        # Run the second MATLAB script
        run_matlab_script(matlab_script_name_2, npics)
    else:
        print(f"Error: MATLAB script '{matlab_script_name_2}' not found in the same folder as the Python script.")
