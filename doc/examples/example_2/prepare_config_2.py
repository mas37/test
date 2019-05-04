def create_gvect_config(filename, gtype):
    with open(filename,'w+') as config:

        config.write('[IO_INFORMATION]'+'\n')
        config.write('#path to the example json'+'\n')
        config.write('input_json_dir = ./examples'+'\n')

        config.write('#path to write the binary'+'\n')
        config.write('output_gvect_dir = ./bin'+'\n')

        config.write('log_dir = ./'+'\n')

        config.write('[SYMMETRY_FUNCTION]'+'\n')
        config.write('#modified Behler-Parrinello'+'\n')
        config.write('type ='+ gtype+'\n')

        config.write('#comma separated strings of atomic symbols'+'\n')
        config.write('species = C'+'\n')

        config.write('#no derivatives'+'\n')
        config.write('include_derivatives = True'+'\n')

        config.write('[PARALLELIZATION]'+'\n')
        config.write('number_of_process = 2'+'\n')

        config.write('[GVECT_PARAMETERS]'+'\n')

        config.write('#The unit of length unit for gvectors parameters'+'\n')
        config.write('#is angstrom. The default can be changed with'+'\n')
        config.write('#gvect_parameters_unit keyword'+'\n')

        config.write('gvect_parameters_unit = angstrom'+'\n')

        config.write('# RADIAL_COMPONENTS'+'\n')

        config.write('eta_rad = 16'+'\n')

        config.write('# radial cutoff'+'\n')
        config.write('Rc_rad = 4.6'+'\n')
 
        config.write('# bias for R_s'+'\n')
        config.write('Rs0_rad = 0.5'+'\n')

        config.write('# number of R_s radial'+'\n')
        config.write('RsN_rad = 16'+'\n')

        config.write('# ANGULAR_COMPONENTS'+'\n')
        config.write('eta_ang = 6.0'+'\n')

        config.write('# angular cutoff'+'\n')
        config.write('Rc_ang = 3.1'+'\n')

        config.write('# bias for angular R_s'+'\n')
        config.write('Rs0_ang = 0.5'+'\n')

        config.write('# number of R_s angular'+'\n')
        config.write('RsN_ang = 8'+'\n')

        config.write('# angular exponent'+'\n')
        config.write('zeta = 50.0'+'\n')

        config.write('# number or theta_s'+'\n')
        config.write('ThetasN = 8'+'\n')
    return None
def create_tfr_config(filename):
    with open(filename,'w+') as config:

        config.write('[IO_INFORMATION]'+'\n')
        config.write('#-- The directory containing the binary gvector files'+'\n')
        config.write('input_dir =./bin'+'\n')


        config.write('# -- The directory where the tfr files will be created'+'\n')
        config.write('output_dir = ./tfr_data'+'\n')

        config.write('#-- How many simulations to store in the same input file'+'\n')
        config.write('elements_per_file = 10'+'\n')
 

        config.write('prefix = tfr_data'+'\n')
        config.write('#-- An optional prefix to add at the beginning of the output files'+'\n')

        config.write('[CONTENT_INFORMATION]'+'\n')
        config.write('# -- Number of species used in the creation of the gvectors'+'\n')
        config.write('n_species = 1'+'\n')
        config.write('include_derivatives = True'+'\n')
    return None


if __name__=='__main__':
    gtype = 'mBP'
    filename_gvect = 'create_gvector.ini'
    filename_tfr = 'create_tfr.ini'
    create_gvect_config(filename_gvect, gtype)
    create_tfr_config(filename_tfr)
