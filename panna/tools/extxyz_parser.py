###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os, sys
import argparse, json
import keys_generator
import multiprocessing as mp
from functools import partial

atom = ['','','','']

#num_processes=18


def parse_exyz(outdir, addhash, xyz_file):
                #print('begin single file parser')
                panna_json = dict()
                panna_json['atoms']=[]
                panna_json['lattice_vectors']=[]
                panna_json['name'] = os.path.splitext( os.path.basename(xyz_file) )[0]
                panna_json['source'] = os.path.abspath(xyz_file)
                # ASSUMPTIONS ABOUT EXTXYZ SOURCE CALCULATION
                panna_json['atomic_position_unit']='cartesian'
                panna_json['unit_of_length']='Angstrom'
                #
                ff = open(xyz_file, 'r')
                lines = ff.readlines()                
                nat = int(lines[0].split()[0])
                # find energy 
                cut = lines[1][lines[1].index('Energy='):]
                cut = cut[cut.index('"')+1:]
                etot = float(cut[:cut.index('"')])
                # unit is assumed to be eV and Angstroms 
                panna_json['energy'] = (etot , 'eV')
                #findLat Lattice ="a11 a12 a13 a21 .."
                cut = lines[1][lines[1].index('Lattice='):]
                cut = cut[cut.index('"')+1:]
                cut = cut[:cut.index('"')].split() 
                panna_json['lattice_vectors'].append( [ float(cut[0]),float(cut[1]),float(cut[2])  ] )
                panna_json['lattice_vectors'].append( [ float(cut[3]),float(cut[4]),float(cut[5])  ] )
                panna_json['lattice_vectors'].append( [ float(cut[6]),float(cut[7]),float(cut[8])  ] ) 
                cut = lines[1][lines[1].index('Properties'):]
                cut = cut[cut.index('=')+1:]
                #cut = cut[:cut.index(' ')].split(':')
                cut = cut.split(':')
                #panna does not need to work with a wide rnage of properties 
                #for the moment all we want is species positions and whether there are forces
                if len(cut) == 6 :
                   #print("Assuming no forces")
                   lforces = False
                   ind = 0
                   for line in lines[2:] :
                       atom[0] = int(ind)
                       ll = line.split()
                       atom[1] = str(ll[0])
                       atom[2] = [float(ll[1]), float(ll[2]), float(ll[3])]
                       panna_json['atoms'].append([atom[0] , atom[1] , atom[2] ])

                elif len(cut) == 9 :
                   #print("Assumes forces")
                   lforces = True 
                   ind = 0
                   for line in lines[2:] :
                       ind = ind+1
                       atom[0] = int(ind)
                       ll = line.split()
                       atom[1] = str(ll[0])
                       atom[2] = [float(ll[1]), float(ll[2]), float(ll[3])]
                       forces =  [float(ll[4]), float(ll[5]), float(ll[6])]
                       panna_json['atoms'].append([atom[0] , atom[1] , atom[2], forces ])

                #
                # Extended XYZ format does not seem to have place for a unique comment etc. 
                # So PANNA adds a unique key to it. 
                panna_json['key'] = 'KeyByPANNA-'+keys_generator.hash_key_v2(panna_json)
                # panna_json['key'] = 'KeyByPANNA-'
                # If desired the name of the json file can be changed with the unique key as below
                if addhash : 
                    panna_json_name = keys_generator.hash_key(panna_json)
                    
                else : 
                    #panna_json_name = f.split('.xyz')[0]
                    panna_json_name = panna_json['name'].split('.xyz')[0]
                #
                    
                with open(outdir.rstrip('/')+"/"+panna_json_name+".example",'w') as outfile:
                    json.dump(panna_json, outfile)
                ff.close()
                return


def main(indir, outdir, addhash, nproc):
    p = mp.Pool(nproc)

    if os.path.isdir(outdir):
       outdir = os.path.abspath(outdir)
       print(outdir)
    else:
       print("outdir not found - making outdir")
       os.makedirs(outdir)
       outdir = os.path.abspath(outdir)
       print(outdir)
    #find files
    xyzfiles = []
    for rt, dirs, files in os.walk(indir):
        for f in files:
            #print(f) 
            if f.endswith('xyz') :
                xyzfiles.append( os.path.join(rt,f) )
    #
    print("num of files found")
    print(len(xyzfiles))
    Parse_exyz = partial(parse_exyz, outdir, addhash)
    i = 0 
    while i <= int(len(xyzfiles) / nproc):
        try: 
            fi = xyzfiles[nproc*i : nproc*(i+1)]
        except IndexError:
            #print('IndexError')
            fi = xyzfiles[nproc*i : len(xyzfiles)]
        data = p.map(Parse_exyz, fi)
        i += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extended XYZ to PANNA json converter')
    parser.add_argument('-i', '--indir', type=str,
                        help='input directory that holds all the xyz files in any subdir structure', required=True)
    parser.add_argument('-o', '--outdir', type=str,
                        help='output directory', required=True)
    parser.add_argument('--addhash', type=bool,  required=False,
                        help='use hash to name jsons', default=False)
    parser.add_argument('--nproc', type=int,
                       help='num threads', required=False, default=1)
    args = parser.parse_args()
    print('begin')
    main(indir=args.indir, outdir=args.outdir, addhash=args.addhash, nproc=args.nproc)
 
