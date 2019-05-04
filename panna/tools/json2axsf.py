import numpy as np
import json
import os
import argparse

def atomic_number(symbol):

    symbols = [ 'H',                               'He',
                'Li','Be', 'B', 'C', 'N', 'O', 'F','Ne',
                'Na','Mg','Al','Si', 'P', 'S','Cl','Ar',
                 'K','Ca','Sc','Ti', 'V','Cr','Mn',
                          'Fe','Co','Ni','Cu','Zn',
                          'Ga','Ge','As','Se','Br','Kr',
                'Rb','Sr', 'Y','Zr','Nb','Mo','Tc',
                          'Ru','Rh','Pd','Ag','Cd',
                          'In','Sn','Sb','Te', 'I','Xe',
                'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd',
                               'Tb','Dy','Ho','Er','Tm','Yb','Lu',
                               'Hf','Ta', 'W','Re','Os',
                          'Ir','Pt','Au','Hg',
                          'Tl','Pb','Bi','Po','At','Rn',
                'Fr','Ra','Ac','Th','Pa',' U','Np','Pu',
                'Am','Cm','Bk','Cf','Es','Fm','Md','No',
                'Lr','Rf','Db','Sg','Bh','Hs','Mt' ]

    return symbols.index(symbol)+1
                   
def json_axsf(indir,outdir):
    ''' This module converts json files to animated xsf file
        for visualization with xcrysden.
        the input files are in .example format.


     indir: the folder that contains the example json.
            indir can be a single file
     outdir: The output folder
             This tool produces a file called animation.axsf in outdir folder   

     The animation step is the number of .example json in indir
    '''


    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if os.path.isfile(indir):
        files=[indir]
    else: 
        files=[os.path.join(indir,x) for x in os.listdir(indir) if x.split('.')[-1]=='simulation']
    Njson=len(files)
    
    dfout=open(os.path.join(outdir,"animation.axsf"),"w")

    dfout.write("ANIMSTEPS"+" " + str(Njson) +"\n")
    steps=1
    for fil in files:
        df = open(fil)
        data = json.load(df)
        lv = data['lattice_vectors']
        print (max((max(lv))))
        print (lv[0] )
        #
        if (not data['lattice_vectors']) or (float(max(max(lv))) < 0.1):
            print('OK!')
            dfout.write("ATOMS"+ " " +str(steps) +"\n")
        else:
            dfout.write("CRYSTALLLL" +"\n") 
            #section
            dfout.write("PRIMVEC"+" " + str(steps) +"\n")
            for latvec in data['lattice_vectors']:
                lvxyz = '{}'.format( ' '.join(map(str, latvec)) )
                dfout.write("  " + str(lvxyz)+"\n")
            #section
            dfout.write("CONVVEC"+" " + str(steps) +"\n")
            for latvec in data['lattice_vectors']:
                lvxyz = '{}'.format( ' '.join(map(str, latvec)) )
                dfout.write("  " + str(lvxyz)+"\n")
            #section
            dfout.write("PRIMCOORD"+ " " +str(steps) +"\n")
            dfout.write(str(len(data['atoms']))+ " 1 "+"\n")
        #
        for atom in data['atoms']:
            #get the atomic number
            try:
                idxx=atomic_number(atom[1])
            except KeyError:
                print('This atomic symbol is not yet')
                print('present in module atomic_number(see above)')
                # print()
                # print('please add it to the module in the format "symbol":atomic number')
                # print('e.g; "Ca":20')
                # exit()
            idx, kind, position, force = atom
            if data['atomic_position_unit']=='crystal':
                position=np.dot(np.transpose(np.asarray(data['lattice_vectors']).astype(float)),
                                      np.asarray(atom[2]).astype(float))
            xyz = '{} {}'.format(str(idxx), ' '.join(map(str, position)) )
            dfout.write(str(xyz)+"\n")
        steps+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Making axsf file from json')
    parser.add_argument('-i', '--indir', type=str,
                        help='in path', required=True)
    parser.add_argument('-o', '--outdir', type=str,
                        help='out path', required=True)
    args = parser.parse_args()
    if args.indir and args.outdir:
       json_axsf(indir=args.indir, outdir=args.outdir)

