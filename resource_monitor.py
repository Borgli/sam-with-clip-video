import argparse
import os.path
import signal

import psutil
import GPUtil
import time
import csv
import subprocess

keep_running = True


def kill_process(sig, frame):
    global keep_running
    keep_running = False


signal.signal(signal.SIGTERM, kill_process)


def get_gpu_temp_and_power():
    try:
        smi_output = subprocess.check_output(['nvidia-smi', '-q', '-d', 'TEMPERATURE,POWER'])
        smi_output = smi_output.decode('utf-8').split('\n')
        temperature_line = [x for x in smi_output if 'GPU Current Temp' in x][0]
        power_line = [x for x in smi_output if 'Power Draw' in x][0]
        temperature = float(temperature_line.split()[-2])
        power = float(power_line.split()[-2])
    except Exception as e:
        print(f"Error getting GPU temperature and power: {e}")
        temperature = power = None
    return temperature, power


def log_system_resources(log_file):
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "CPU Usage(%)", "Memory Usage(%)", "Memory Bytes", "GPU Usage", "GPU Memory Usage", "Disk Usage(%)", "Bytes Sent", "Bytes Recv", "GPU Temperature", "GPU Power Usage"])

        # Get initial network stats
        net_io_start = psutil.net_io_counters()

        while keep_running:
            # Get current time
            current_time = time.ctime()

            # Get CPU usage
            cpu_usage = psutil.cpu_percent()

            # Get CPU temp
            #cpu_average_temp = psutil.sensors_temperatures()

            # Get CPU power

            # Get Memory usage
            memory_usage = psutil.virtual_memory().percent
            memory_usage_used = psutil.virtual_memory().used

            # Get GPU usage
            GPUs = GPUtil.getGPUs()
            gpu_usage = GPUs[0].load if GPUs else 0
            gpu_memory_usage = GPUs[0].memoryUtil if GPUs else 0

            # Get Disk usage
            disk_usage = psutil.disk_usage('/').percent

            # Get Network usage (bytes sent and received since last check)
            net_io_end = psutil.net_io_counters()
            bytes_sent = net_io_end.bytes_sent - net_io_start.bytes_sent
            bytes_recv = net_io_end.bytes_recv - net_io_start.bytes_recv
            net_io_start = net_io_end

            # Get GPU temperature and power usage
            gpu_temp, gpu_power = get_gpu_temp_and_power()

            # Write to CSV file
            writer.writerow([current_time, cpu_usage, memory_usage, memory_usage_used, gpu_usage, gpu_memory_usage, disk_usage, bytes_sent, bytes_recv, gpu_temp, gpu_power])

            # Wait for a while before getting the next reading
            time.sleep(1)


if __name__ == '__main__':
    print("Hello world")
    parser = argparse.ArgumentParser(prog="Resource Monitor",
                                     description="Monitors the system resources and saves them to the file "
                                                 "'resource_log.csv'. A folder can be given and it will then save it"
                                                 "to that folder.")
    parser.add_argument('--log_dir', default="./")
    args = parser.parse_args()
    log_system_resources(os.path.join(args.log_dir, 'resource_log.csv'))
