###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
#QE xml -> panna json
#
#Looks inside the input directory and finds all files that end with .xml 
#Since the name of this file is given to the name of the generated json, 
#it avoids using the "data-file-schema.xml"s that are in the .save directories 
#and just uses the <prefix>.xml copy outside the .save copy. 
#This behavior can be easily changed based on demand
#
from xml.etree import ElementTree as et
import os, sys
import argparse
import json
# 
import keys_generator
#uses hash_key

atom = ['','','','']

def main(indir, outdir, addhash, **kvargs):
    if os.path.isdir(outdir): 
       outdir = os.path.abspath(outdir)
       print(outdir)
    else:
       os.mkdir(outdir)
       outdir = os.path.abspath(outdir)
       print(outdir)
    #find QE xmls with prefix names
    for rt, dirs, files in os.walk(indir):
        for f in files:
            #print(f) 
            if f.endswith('xml') and f != "data-file-schema.xml" :
                #initialize json dictionary
                panna_json = dict()
                panna_json['atoms']=[]
                #
                xml_file = os.path.join(rt,f)
                #panna_json['name'] = os.path.splitext( os.path.basename(xml_file) )[0]
                panna_json['source'] = os.path.abspath(xml_file)
                #is this a valid xml file?
                try: 
                    tree = et.parse(xml_file)
                except et.ParseError as err:
                    print("\nError parsing XML file:  {}".format(f))
                    #sys.exit(1)
                root = tree.getroot()
                #INPUT
                inp = root.find('input')
                control_variables = inp.find('control_variables')
                #whether forces are calculated
                lforces = control_variables.find('forces').text
                panna_json['key'] = control_variables.find('prefix').text
                #emine: in the future key or json filename might become the hash of the xml file maybe to ensure uniqueness ?
                atomic_str = inp.find('atomic_structure')
                # alat = atomic_str.attrib['alat'] - panna doesnt use
                nat = int(atomic_str.attrib['nat']) # used to fill in forces when they are not calculated
                # ATOMIC POSITIONS
                atomic_pos = atomic_str.find('atomic_positions')
                #for idx, at in enumerate(atomic_pos.findall('atom')) :
                #    atom[0] = int(at.attrib['index'])
                #    atom[1] = str(at.attrib['name'])
                #    atom[2] = at.text.split() # scientific notation atomic positions list
                #    atom[2] = [float(atom[2][0]),float(atom[2][1]),float(atom[2][2])]
                #    panna_json['atoms'].append([atom[0] , atom[1] , atom[2], 'forces here' ])
                # QE defaults - unchecked
                panna_json['atomic_position_unit']='cartesian'
                panna_json['unit_of_length']='bohr'
                # CELL
                cell = atomic_str.find('cell')
                panna_json['lattice_vectors']=[]
                for vector in cell :
                    panna_json['lattice_vectors'].append([float(vector.text.split()[0]),float(vector.text.split()[1]),float(vector.text.split()[2])])
                # INFO FROM OUTPUT
                outp = root.find('output')
                # TOTAL ENERGY
                total_energy = outp.find('total_energy')
                etot = float(total_energy.find('etot').text)
                unit_of_energy = "Ha" #QE default - any reason to change this?
                panna_json['energy']=(etot,unit_of_energy)
                # FORCES
                # assues that forces are calculated
                forces = []
                if lforces == 'true' :
                    force_array = outp.find('forces').text.split()
                    for i in range(0,len(force_array),3):
                        forces.append( [float(force_array[i]),float(force_array[i+1]), float(force_array[i+2])] )
                #else: 
                #    for i in range(0,nat):
                #        forces.append( )

                # ATOMIC POSITIONS -better added here than beforei
                if lforces == 'true' :
                    for idx, at in enumerate(atomic_pos.findall('atom')) :
                        atom[0] = int(at.attrib['index'])
                        atom[1] = str(at.attrib['name'])
                        atom[2] = at.text.split() # scientific notation atomic positions list
                        atom[2] = [float(atom[2][0]),float(atom[2][1]),float(atom[2][2])]
                        panna_json['atoms'].append([atom[0] , atom[1] , atom[2], forces[idx] ])
                elif lforces == 'false' :
                    for idx, at in enumerate(atomic_pos.findall('atom')) :
                        atom[0] = int(at.attrib['index'])
                        atom[1] = str(at.attrib['name'])
                        atom[2] = at.text.split() # scientific notation atomic positions list
                        atom[2] = [float(atom[2][0]),float(atom[2][1]),float(atom[2][2])]
                        panna_json['atoms'].append([atom[0] , atom[1] , atom[2] ])

    
                #print panna_json
                #use hash if necessary
                if addhash : 
                    panna_json_name = keys_generator.hash_key_v2(panna_json)
                else : 
                    panna_json_name = f.split('.xml')[0]
                #
                #cwd = os.getcwd()
                #if not os.path.exists(cwd+"/"+outdir):
                #    os.makedirs(outdir)
                #print(outdir)
                #remove the trailing slash if there was any
                with open(outdir.rstrip('/')+"/"+panna_json_name+".example",'w') as outfile:
                    json.dump(panna_json, outfile)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QE xml to PANNA json converter')
    parser.add_argument('-i', '--indir', type=str,
                        help='input directory that holds all the outdirs', required=True)
    parser.add_argument('-o', '--outdir', type=str,
                        help='output directory', required=True)
    parser.add_argument('--addhash', action='store_true', dest='addhash',
                        help='use hash to name jsons')
    args = parser.parse_args()
   
    main(args.indir, args.outdir, args.addhash)
    
    #main(args.indir, args.outdir)



