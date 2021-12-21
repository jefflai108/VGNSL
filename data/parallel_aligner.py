import json 
import argparse 
from tqdm import tqdm
import os
from pathlib import Path
import subprocess
from collections import defaultdict 

def parallel_run(args): 
    spk_list = collect_speaker(args.data_directory) 
    interval = len(spk_list) // args.jobs
    tgt_spk_list = spk_list[args.job_partition * interval:(args.job_partition+1) * interval+1]
    
    print('aligning %d speakers in total' % len(tgt_spk_list))
    lexicon_path = os.path.join(args.corpus_directory, 'librispeech-lexicon.txt')
    already_processed_speakers = _read_file(args.done_list)
    for spk in tqdm(tgt_spk_list): 
        if spk in already_processed_speakers: 
            print('skipping speaker %s' % spk)
            continue 
        print('aligning speaker %s' % spk)
        dump_dir = os.path.join(args.corpus_directory, '.mfa_adapt-' + spk)
        spk_dir = os.path.join(args.data_directory, spk)
        aligned_spk_dir = spk_dir.replace('/wavs-speaker/', '/wavs-speaker-aligned/')
        bashCommand = "mfa adapt " + spk_dir + " " + lexicon_path + " english " + aligned_spk_dir + " -t " + dump_dir + " -j 10 -v --debug --clean --beam 1000 --retry-beam=1500" # set beam to 1k
        print(bashCommand)
        execute(bashCommand.split(), args.done_list, spk)

def execute(cmd, speaker_done_list, spk):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for line in popen.stdout:
        print(line, end='')
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    else: 
        _write_to_file(spk, speaker_done_list) # write to `done_alignment_speakers_list2.txt` to keep track 

def collect_speaker(data_directory): 
    spk_list = []
    for filename in sorted(os.listdir(data_directory)): 
        spk_list.append(filename)

    return spk_list

def _read_file(fpath): 
    with open(fpath, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]

    return content 

def _write_to_file(string, fpath): 
    f = open(fpath, 'a') 
    f.write(string + '\n')
    f.close()

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-directory', '-c', type=str)
    parser.add_argument('--data-directory', '-d', type=str)
    parser.add_argument('--jobs', '-N', type=int)
    parser.add_argument('--job-partition', '-n', type=int)
    parser.add_argument('--done-list', '-l', type=str)
    args = parser.parse_args()

    parallel_run(args)
