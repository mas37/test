def activation(act):
    if act == 1:
        act_funct = 'gaussian' 
    if act == 2:
        act_funct = 'rbf' 
    if act == 3:
        act_funct = 'relu' 

    return act_funct


def train_config(filename, act):
    act_funct = activation(act)
    with open(filename,'w+') as config:
        
        config.write('[IO_INFORMATION]'+'\n')

        config.write('data_dir = ./tfr_data/training_set'+'\n')
        config.write('train_dir = ./train_' + act_funct+'\n')

        config.write('log_frequency = 100'+'\n')
        config.write('save_checkpoint_steps = 1000'+'\n')
        config.write('max_ckpt_to_keep = 250000'+'\n')

        config.write('[DATA_INFORMATION]'+'\n')
        config.write('atomic_sequence = C'+'\n')
        # offset in order of the atomic_sequence in eV
        config.write('output_offset =-245.667607369859'+'\n')

        config.write('[TRAINING_PARAMETERS]'+'\n')
        config.write('batch_size = 20'+'\n')
        config.write('learning_rate = 0.01'+'\n')
        config.write('max_steps = 50000'+'\n')
        config.write('forces_cost = 0.02'+'\n')
        config.write('learning_rate_constant = True'+'\n')
        config.write('learning_rate_decay_factor = 0.95'+'\n')
        config.write('learning_rate_decay_step = 1000'+'\n')

        config.write('[DEFAULT_NETWORK]'+'\n')
        config.write('g_size = 80'+'\n')
        config.write('architecture = 32:32:1'+'\n')
        config.write('trainable = 1:1:1'+'\n')
        config.write('activations = '+str(act)+':'+str(act)+':0'+'\n')

        config.write('[PARALLELIZATION]'+'\n')
        config.write('dataset_cache = False'+'\n')
        config.write('#PANNA defaults are used here'+'\n')

    return None

def evaluate_config(filename, act):
    act_funct = activation(act)
    with open(filename,'w+') as config:

        config.write('[IO_INFORMATION]'+'\n')

        config.write('data_dir = ./tfr_data/validation_set'+'\n')
        config.write('train_dir = ./train_' + act_funct+'\n')
        config.write('eval_dir = ./eval_' + act_funct+'\n')

        config.write('[TFR_STRUCTURE]'+'\n')
        config.write('g_size = 80'+'\n')

        config.write('[VALIDATION_OPTIONS]'+'\n')
        config.write('single_step = True'+'\n')
        config.write('compute_forces = True'+'\n')
    return None

if __name__== '__main__':

    for act in range(1,4):
        act_funct = activation(act)
        filename_t = 'train_'+ act_funct+'.ini'
        filename_e = 'validation_'+ act_funct+'.ini'
        train_config(filename_t, act)
        evaluate_config(filename_e, act)
        print(act_funct, 'done!!!!')
