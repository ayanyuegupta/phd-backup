import subprocess
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = args.a_s
    
    as_args = ['-a_s', f'{a_s}']
    subprocess.run(['python', 'add_centroids.py'] + as_args)
    subprocess.run(['python', 'add_senses.py'] + as_args)
    subprocess.run(['python', 'analyse_sense3.py'] + as_args)

if __name__ == '__main__':
    main()

