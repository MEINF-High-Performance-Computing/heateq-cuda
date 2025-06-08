import subprocess
import re
import os
import time

# Control variables
compile_code = False  # Set to False to skip compilation
delete_previous = True  # Set to False to keep previous output files
run_benchmarks = True  # Set to False to skip running benchmarks
run_tests = False  # Set to False to skip running tests
create_csv = True  # Set to False to skip CSV creation

# Paths
filename = "heat_cuda"
cuda_file = f"{filename}.cu"
executable = f"./{filename}"

bmp_folder = "bmp"
results_folder = "results"

serial_bmp_folder = "bmp_serial"

sizes = [100, 1_000, 2_000]
steps = [100, 1_000, 10_000, 100_000]

cuda_threads = [
                (2, 1), # 2 threads in total
                (2, 2), # 4 threads in total
                (4, 2), # 8 threads in total
                (4, 4), # 16 threads in total
                (8, 4), # 32 threads in total
                (8, 8), # 64 threads in total
                (16, 8), # 128 threads in total
                (16, 16), # 256 threads in total
                (32, 16), # 512 threads in total
                (32, 32) # 1024 threads in total
            ]

def compile_program():
    try:
        # Compile the CUDA file
        compile_command = f"nvcc {cuda_file} -o {filename}"
        subprocess.run(compile_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}")
        raise
    
def setup_folders(delete=False):
    os.makedirs(bmp_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    if delete:
        # Clean up previous output files
        subprocess.run(f"rm -f {bmp_folder}/*.bmp", shell=True)
        subprocess.run(f"rm -f {results_folder}/*.txt", shell=True)

def execute_benchmarks():
    for size in sizes:
        for step in steps:
            for threads in cuda_threads:
                thread_x, thread_y = threads
                bmp_output_file = f"output_cuda_nx_{size}_st_{step}_thx_{thread_x}_thy_{thread_y}_th_{thread_x*thread_y}.bmp"
                output_file = f"heat_cuda_nx_{size}_st_{step}_thx_{thread_x}_thy_{thread_y}_th_{thread_x*thread_y}.txt"
                command = f"{executable} {size} {step} {os.path.join(bmp_folder, bmp_output_file)} {thread_x} {thread_y}"
                print(f"Running: {command}")
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                
                # Extract execution time using regex
                match = re.search(r"The Execution Time=\s*([\d.]+)", result.stdout)
                if match:
                    exec_time = match.group(1)
                    print(f"Execution time: {exec_time} s")

                    # Save to file
                    with open(os.path.join(results_folder, output_file), "w") as f:
                        f.write(f"{exec_time}\n")
                else:
                    print("Execution time not found in output.")
                    print("Program output:")
                    print(result.stdout)
                    print("Error output:")
                    print(result.stderr)
                time.sleep(1)  # import time
    print("All tests completed successfully.")
    
def check_images():
    for size in sizes:
        for step in steps:
            for threads in cuda_threads:
                thread_x, thread_y = threads
                bmp_output_file_cuda = f"output_cuda_nx_{size}_st_{step}_thx_{thread_x}_thy_{thread_y}_th_{thread_x*thread_y}.bmp"
                bmp_output_file_serial = f"output_serial_nx_{size}_st_{step}.bmp"
                bmp_path_cuda = os.path.join(bmp_folder, bmp_output_file_cuda)
                bmp_path_serial = os.path.join(serial_bmp_folder, bmp_output_file_serial)
                result = subprocess.run(f"cmp -s {bmp_path_cuda} {bmp_path_serial}", text=True, capture_output=True, shell=True)
                if result.returncode == 0:
                    print(f"{bmp_output_file_cuda} --> OK")
                else:
                    print(f"FAILURE: {bmp_output_file_cuda} does not match {bmp_output_file_serial}")
                    

def write_csv():
    csv_file = "heat_results.csv"
    with open(csv_file, "w+") as f:
        f.write("config,size,step,threads,time\n")
        for size in sizes:
            for threads in cuda_threads:
                thread_x, thread_y = threads
                for step in steps:
                    output_file = f"heat_cuda_nx_{size}_st_{step}_thx_{thread_x}_thy_{thread_y}_th_{thread_x*thread_y}.txt"
                    try:
                        with open(os.path.join(results_folder, output_file), "r") as result_file:
                            exec_time = result_file.read().strip()
                            f.write(f"CUDA,{size},{step},{thread_x*thread_y},{exec_time}\n")
                    except FileNotFoundError:
                        print(f"File {output_file} not found. Skipping.")

if __name__ == "__main__":
    try:
        if compile_code:
            compile_program()
            
        setup_folders(delete=delete_previous)
        
        if run_benchmarks:
            execute_benchmarks()
            
        if run_tests:
            check_images()
        
        if create_csv:
            write_csv()


    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e.cmd}")
        print(f"Error message: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
# end main