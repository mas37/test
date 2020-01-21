import numpy as np
import os
import json
import argparse
ry2ev=13.6056980659
b2a=0.529177

def make_json(indir,outdir):
    '''make json from stdout of QE.

       This routine reads and convert single QE scf standard output to
       json. Th scf out put files contained in indir must have .out extension
       No check is made. If the output has no complete forces, the json for
       that calculation is skipped. The output json is completely random but has
       information about the energy. The name of json is same as the qe output file
       with all '.' removed except that the .example extension is added at the end

       indir: folder tha contains the output files
       outdir: folder where output are written

    '''

    if not os.path.exists(outdir):
        os.mkdir(outdir)


    files = [os.path.join(indir,z) for z in os.listdir(indir) if z.split('.')[-1] == 'out']
    for f in files:
        pos_id = f.split('/')[-1].split('.')[0]
        with open(f,'r') as df:
            lines = df.readlines()

        config_info = dict()
        config_info['lattice_vectors'] = []
        config_info['energy'] = []
        config_info['atomic_position_unit'] = 'cartesian'
        config_info['unit_of_length'] = 'bohr'


        pos = []
        forces = []
        for i in range(len(lines)): 
            l = lines[i].split()
            if 'lattice' in l and 'parameter' in l: 
                #celpar in bohr
                alat = float(l[4])
            #number of atoms
            if 'atoms/cell' in l:
                Natoms = int(l[4]) 

           
            #lattice vectors
            if 'a(1)' in l:
                config_info['lattice_vectors'].append([float(l[3])*alat,float(l[4])*alat,float(l[5])*alat])  
            if 'a(2)' in l:
                config_info['lattice_vectors'].append([float(l[3])*alat,float(l[4])*alat,float(l[5])*alat])  
            if 'a(3)' in l:
                config_info['lattice_vectors'].append([float(l[3])*alat,float(l[4])*alat,float(l[5])*alat])  
            #atoms
            if 'positions' in l:
                for j in range(1,Natoms+1):
                    line = lines[i+j].split()
                    idx = int(line[0])
                    kind = line[1]
                    x = float(line[6])*alat
                    y = float(line[7])*alat
                    z = float(line[8])*alat
                    pos.append([idx,kind,[x,y,z]])
           
            #energy     
            if '!' in l: 
                en = float(l[4])
                config_info['energy'].append(en)
                config_info['energy'].append("Ry")
  
            #get forces
            if  'atom' in l and 'force' in l:
                forces.append([float(l[6]),float(l[7]),float(l[8])])
                fx = float(l[6])
                fy = float(l[7])
                fz = float(l[8])


        for i in range(len(pos)):
            if len(pos) == len(forces):
                pos[i].append(forces[i])
        config_info['atoms'] = pos
        if len(config_info['energy']) > 1 and len(forces) == len(pos):

            config_info = json.dumps(config_info)
 
            with open(os.path.join(outdir,f.split('/')[-1].replace('.','')+'.example'),'w+') as outfile:
                outfile.write(config_info)
                
        
        else:
            print(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creating json')
    parser.add_argument('-i', '--indir', type=str,
                        help='input dir', required=True)
    parser.add_argument('-o', '--outdir', type=str,
                        help='out path', required=True)
    args = parser.parse_args()
    if args.indir and args.outdir:
       make_json(indir=args.indir, outdir=args.outdir)
              
