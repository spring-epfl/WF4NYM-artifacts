#!/usr/bin/env python
# coding: utf-8

import os
import re
import argparse

# Regular expression pattern to match tcpdump output
TCP_DUMP_PATTERN = re.compile(
    r"(\d+)\.(\d{6}).*length (\d+)\) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\.(\d{1,5}).* > (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\.(\d{1,5}).* tcp (\d{1,5})"
)

def tcp_dump(input_file, output_folder):
    """
    Extract tcpdump data from pcap file and save to a text file.

    Parameters:
    - input_file: str, path to the input pcap file
    - output_folder: str, path to the output folder
    """
    split_path = input_file.split("/")
    url_path = os.path.join(output_folder, split_path[-3])
    os.makedirs(url_path, exist_ok=True)
    output_file = os.path.join(url_path, split_path[-2].zfill(2) + ".tcpdump")
    os.system(
        """tcpdump -r "{0}" -n -l -tt -q -v | sed -e 's/^[ 	]*//' | grep -i tcp | awk '/length ([0-9][0-9]*)/{{printf "%s ",$0;next}}{{print}}' > '{1}'""".format(
            input_file, output_file
        )
    )

class Packet:
    def __init__(self, timestamp=0, packetsize=0, src_ip="", dst_ip=""):
        self.timestamp = float(timestamp)/1000
        self.packetsize = int(packetsize)
        self.src_ip = src_ip
        self.dst_ip = dst_ip

    def __str__(self):
        return f"{self.timestamp}:{self.packetsize}"

class Stream:
    def __init__(self, url=""):
        self.url = url
        self.packets = []

    def __str__(self):
        return f"{self.url} {len(self.packets)} {' '.join(map(str, self.packets))}"

def get_file_name(stream, output_folder):
    """
    Generate a file name for the stream data.

    Parameters:
    - stream: Stream object
    - output_folder: str, path to the output folder

    Returns:
    - str, path to the output file
    """
    outputfilename = (
        stream.url.replace("/", "_")
        .replace(":", "_")
        .replace("?", "_")
        .replace("﻿", "")
    )
    return os.path.join(output_folder, outputfilename[:100])

def print_stream(stream, output_folder):
    """
    Save the stream data to a file.

    Parameters:
    - stream: Stream object
    - output_folder: str, path to the output folder
    """
    with open(get_file_name(stream, output_folder), "a") as outputfile:
        outputfile.write(str(stream) + "\n")

def urls_to_streams(url, runfiles, src_ip, dst_ips, no_dst_filter, output_folder, ignore_port_9002=False):
    """
    Convert tcpdump data to stream objects.

    Parameters:
    - url: str, base URL
    - runfiles: list, list of tcpdump files
    - src_ip: list, list of source IPs
    - dst_ips: list, list of destination IPs
    - no_dst_filter: bool, whether to apply destination IP filter
    - output_folder: str, path to the output folder
    - ignore_port_9002: bool, whether to exclude port 9002 packets

    Returns:
    - list, list of Stream objects
    """
    streams = []
    src_ip = [src_ip]
    for current_file in runfiles:
        current_stream = Stream(url.split("/")[-1])
        if os.path.exists(get_file_name(current_stream, output_folder)):
            continue
        with open(current_file, "r") as tcpdumpfile:
            for tcpdumpline in tcpdumpfile:
                match = TCP_DUMP_PATTERN.search(tcpdumpline)
                if match is None:
                    print("Malformed line:" + tcpdumpline)
                    continue

                if match.group(5) == "80" or match.group(7) == "80" or match.group(5) == "8081" or match.group(7) == "8081":
                    continue

                # Skip port 9002 packets if ignore_port_9002 is True
                if ignore_port_9002 and (match.group(5) == "9002" or match.group(7) == "9002"):
                    continue

                if match.group(8) == "0":
                    continue

                packet = Packet(
                    match.group(1) + match.group(2),
                    match.group(8),
                    match.group(4),
                    match.group(6),
                )
                
                if no_dst_filter and packet.src_ip not in src_ip and packet.dst_ip not in src_ip:
                    continue
                if not no_dst_filter and not (
                    (packet.src_ip in src_ip and packet.dst_ip in dst_ips)
                    or (packet.src_ip in dst_ips and packet.dst_ip in src_ip)
                ):
                    continue

                if packet.src_ip in src_ip:
                    packet.packetsize = -1 * packet.packetsize

                current_stream.packets.append(packet)

        if len(current_stream.packets) > 0:
            streams.append(current_stream)
    return streams


def create_tcp(output_dataset, src_ip, dst_ips, no_dst_filter, ignore_port_9002=False):
    output_folder = os.path.join(output_dataset, "output-tcp")
    
    os.makedirs(output_folder, exist_ok=True)
    urls = [ url.path for url in os.scandir(output_dataset) if url.is_dir() and "output-tcp" not in url.path]
    for url in urls:
        #print(url)
        runfiles = []
        for root, dirs, files in os.walk(url):
            for file in files:
                if file.endswith('.tcpdump') and ".ipynb_checkpoints" not in root:
                    runfiles.append(os.path.join(root, file))
        runfiles.sort()
        #print(runfiles)
        streams = urls_to_streams(url, runfiles, src_ip, dst_ips, no_dst_filter, output_folder, ignore_port_9002)
        
        for index, stream in enumerate(streams):
            print_stream(stream, output_folder)


def find_unique_filename(base_path, base_filename):
    # Start with no suffix
    output_file = os.path.join(base_path, base_filename + ".tcpdump")
    i = 0
    # Loop until a unique filename is found
    while os.path.isfile(output_file):
        i += 1
        output_file = os.path.join(base_path, base_filename + str(i) + ".tcpdump")
    return output_file


def process_data(input_path, output_path, datasets, src_ip, dst_ips, no_dst_filter, ignore_port_9002=False):
    """
    Process pcap files and extract tcpdump data.

    Parameters:
    - input_path: str, path to the input dataset folder
    - output_path: str, path to the output dataset folder
    - datasets: list, list of dataset folders
    - src_ip: str, source IP
    - dst_ips: list, list of destination IPs
    - no_dst_filter: bool, whether to apply destination IP filter
    - ignore_port_9002: bool, whether to exclude port 9002 packets
    """
    for dataset in datasets:
        for path, subdirs, files in os.walk(os.path.join(input_path, dataset)):
            for name in files:
                if "capture.pcap" in name and path.endswith("_0"):
                    input_file = os.path.join(path, name)
                    split_path = input_file.split("/")
                    url_path = os.path.join(output_path, split_path[-3])
                    #output_file = os.path.join(url_path, split_path[-2].zfill(2) + ".tcpdump")
                    base_filename = split_path[-2].zfill(2)
                    output_file = find_unique_filename(url_path, base_filename)
                    #if not os.path.isfile(output_file):
                    tcp_dump(input_file, output_path)
    create_tcp(output_path, src_ip, dst_ips, no_dst_filter, ignore_port_9002)

def main():
    """
    Main function to parse arguments and start data processing.
    """
    parser = argparse.ArgumentParser(description="Process TCP dump data.")
    parser.add_argument("input_path", type=str, help="Input path")
    parser.add_argument("output_path", type=str, help="Output path")
    parser.add_argument("datasets", nargs='+', help="List of datasets")
    parser.add_argument("--src_ip", type=str, default="10.1.1.1", help="Source IP")
    parser.add_argument("--dst_ips", nargs='+', default=["139.162.200.242"], help="List of destination IPs")
    parser.add_argument("--no_dst_filter", action="store_true", default=True, help="No destination filter")
    parser.add_argument("--ignore_port_9002", action="store_true", help="Exclude packets with port 9002 as source or destination")
    
    args = parser.parse_args()
    
    process_data(args.input_path, args.output_path, args.datasets, args.src_ip, args.dst_ips, args.no_dst_filter, args.ignore_port_9002)

if __name__ == "__main__":
    main()
