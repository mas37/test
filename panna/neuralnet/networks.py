###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import tensorflow as tf
import numpy as np

from neuralnet.layers import hidden_layer


# All to all connected, feedforward network.
def network_A2A(batch_of_species,
                batch_of_gvects,
                layer_size,
                trainability,
                activations,
                gvect_size,
                batch_size,
                Nspecies,
                atomic_label,
                import_layer=None,
                compute_gradients=False,
                reuse=None):
    """ New network with variable architecture annd
        species resolved weights.

    Args:
      batch_of_species: [batch size x max number of atoms per molecule]
                      species vector for each element of the batch
      batch_of_gvects: batch of gvectors

      layer_size : a list of lists with the size of each
                     hidden layer, for each species
      trainability : a list of lists with the boolean flag
                         of the trainable status of each
                         hidden layer, for each species
      activations : list of activations - integer -see parser_callable
                         for their list
      gvect_size : gvector size
      batch_size : number of calculations in a batch
      Nspecies : number of species
      import_layer : a list of lists of tuple,
                     each tuple has 2 elements,
                     weights and biases
                     that can be either a tensor with
                     correct shape or None
      reuse: whether to reuse variables with the same name


    Returns:
      Energy: tf.tensor of energies
      natoms_batch: tf.tensor of number of atoms for each element of the
                    batch
    """
    #Species sorted list of lists [specie 1 list ,specie 2 list,..]:
    #atomic energies for all the atoms in the batch, species sorted
    atomic_energies = []
    #for each atom, the molecule number in the batch, to keep track of where they came from, species sorted
    molecule_index = []

    # List of derivatives wrt Gs, needed for differentiation
    dEdG_list = []
    # create a network for each species
    for s in range(Nspecies):
        # copy locally the layer structure for this species
        lsize_list = layer_size[s].copy()
        # first layer is gvects for each species
        lsize_list.insert(0, gvect_size)
        # now layer_size also holds the size of output layer
        Nlayers = len(lsize_list) - 1

        # in the batch of species matrix [batch size x max number of atoms],
        # get the index of atoms of this species
        # atomic index [number of atoms_of_this_species x 2]
        atomic_index = tf.where(tf.equal(batch_of_species, s))

        # Select the gvects of the atoms of this species
        NN_layers_list = [tf.gather_nd(batch_of_gvects, atomic_index)]
        if compute_gradients:
            NN_d_layers_list = [tf.ones(tf.shape(NN_layers_list[0]))]

        # Keep track of which molecule this atom came from
        # by using the row position in the batch of species matrix
        # row position is the order of that molecule in the given batch.
        molecule_index.append(atomic_index[:, 0])

        # Now build the neural network for this species
        # (looping on number of hidden layers for this species)
        # First all to all connected section with gaussian activation
        for l in range(Nlayers):
            with tf.variable_scope(
                    "species_{}_layer_{}".format(s, l + 1), reuse=reuse):
                init_values = import_layer[s][l] if import_layer else (None,
                                                                       None)
                NN_layers_list.append(
                    hidden_layer(
                        NN_layers_list[l],
                        in_size=lsize_list[l],
                        out_size=lsize_list[l + 1],
                        trainable=trainability[s][l],
                        activation=activations[s][l],
                        init_values=init_values))
        # if the last layer is linear (as it has to be right now)
        # the end result is [number of atoms of this species , 1]
        # reshaping to flatten it to 1D; [number of atoms of this species]
        # (in the most common form the output will be energies, so naming it as such.)
        atomic_output = NN_layers_list[Nlayers]
        atomic_energies.append(tf.reshape(atomic_output, [-1]))
        if compute_gradients:
            dEdG_list.append(tf.gradients(atomic_output, NN_layers_list[0])[0])

    # end of loop over species

    # Sum the atomic energies for each molecule, add relevant data to tensorflow board
    energies = tf.zeros(batch_size)
    batch_of_natoms = tf.zeros(batch_size)
    for s in range(Nspecies):
        # Avoid loop over molecules by creating a one-hot matrix:
        # Remember molecule_index is the list of molecules the atoms of this species came from
        # this mask is then a matrix of ( batchsize x len(molecule index[s] )
        molecule_index_mask = tf.one_hot(molecule_index[s], depth=batch_size)
        # tile the atomic energies[s] such that atomic energies are repeated batch_size times
        atomic_energies_tiled = tf.tile(tf.expand_dims(
                                        atomic_energies[s],-1), \
                                        [1, batch_size])
        # multiply the mask with the tiled atomic energy matrices element by element
        # such that when reduced over the first dimension (axis=0)
        # what remains is the total energy contribution of this species to each molecule
        s_energies = tf.reduce_sum(
            molecule_index_mask * atomic_energies_tiled, axis=0)

        mean, var = tf.nn.moments(atomic_energies[s], axes=[0])

        tf.summary.scalar("2.Mean_energy/S{}.{}".format(s, atomic_label[s]),
                          mean)
        tf.summary.scalar("3.Std_energy/S{}.{}".format(s, atomic_label[s]),
                          tf.sqrt(var))

        energies = energies + s_energies

        s_natoms = tf.reduce_sum(molecule_index_mask, axis=0)
        tf.summary.scalar("4.N_atoms/S{}.{}".format(s, atomic_label[s]),
                          tf.reduce_sum(s_natoms))
        batch_of_natoms = batch_of_natoms + s_natoms

    for x in tf.get_collection(tf.GraphKeys.WEIGHTS):
        name = x.op.name.split("/")[0]
        nameparts = name.split("_")
        tf.summary.histogram(
            "S{}.{}/W{}".format(nameparts[1], atomic_label[(int)(
                nameparts[1])], nameparts[3]), x)
    for x in tf.get_collection(tf.GraphKeys.BIASES):
        name = x.op.name.split("/")[0]
        nameparts = name.split("_")
        tf.summary.histogram(
            "S{}.{}/b{}".format(nameparts[1], atomic_label[(int)(
                nameparts[1])], nameparts[3]), x)

    if compute_gradients:
        return energies, batch_of_natoms, dEdG_list

    return energies, batch_of_natoms


def loss_NN(batch_energies, batch_energies_dft, batch_natoms,
            loss_func="quad"):
    """ this is simply our cost function

    Args:
        batch_energies = prediction of the network
        batch_energies_dft = energies labels
        batch_natoms = number of atoms
        loss_func = name of the loss function:
          - quad
          - exp_quad
          - quad_atom
          - exp_quad_atom

    Returns:
        the loss value
        tensor with delta_e for each element of the batch

    """

    debug = False

    with tf.name_scope('loss_calculation'):
        # reshape for convenience[100,1]->[100]
        batch_energies_dft = tf.reshape(
            batch_energies_dft, [-1], name='reshape_dft_en')

        # loss function
        batch_delta_e = (batch_energies - batch_energies_dft)
        batch_delta_e2 = tf.square(batch_delta_e)
        if debug:
            batch_delta_e = tf.Print(
                batch_delta_e,
                data=[batch_delta_e],
                summarize=1024,
                message='batch_delta_e:')
        if loss_func == "quad":
            tot_loss = tf.reduce_sum(batch_delta_e2, name='1.LOSS-Delta_E2')
        elif loss_func == "exp_quad":
            tot_loss = 0.5 * tf.exp(
                2.0 * tf.reduce_sum(batch_delta_e2),
                name='1.LOSS-Exp_Delta_E2')
        elif loss_func == "quad_atom":
            tot_loss = tf.reduce_sum(
                tf.div(batch_delta_e2, tf.square(batch_natoms)),
                name='1.LOSS-Delta_E2_div_Natom2')
        elif loss_func == "quad_std_atom":
            tot_loss = tf.reduce_sum(
                tf.div(batch_delta_e2, batch_natoms),
                name='1.LOSS-Delta_E2_div_Natom')
        elif loss_func == "exp_quad_atom":
            tot_loss = 0.5 * tf.exp(
                2.0 * tf.reduce_sum(
                    tf.div(batch_delta_e2, tf.square(batch_natoms))),
                name='1.LOSS-Exp_Delta_E2_div_Natom2')
        elif loss_func == "exp_quad_std_atom":
            tot_loss = 0.5 * tf.exp(
                2.0 * tf.reduce_sum(
                    tf.div(batch_delta_e2, batch_natoms)),
                name='1.LOSS-Exp_Delta_E2_div_Natom')
        elif loss_func == "quad_exp_tanh_atom":
            a = tf.constant(5.0)
            red_sum = tf.div(
                tf.reduce_sum(tf.div(batch_delta_e2, tf.square(batch_natoms))),
                a)
            tot_loss = tf.add(tf.reduce_sum(batch_delta_e2), tf.exp(tf.multiply(a, tf.tanh(red_sum))), \
                              name = '1.LOSS-quad_exp_tanh_Delta_E2_div_Natom2')
        elif loss_func == "quad_exp_tanh_std_atom":            
            a = tf.constant(5.0)
            red_sum = tf.div(
                tf.reduce_sum(tf.div(batch_delta_e2, batch_natoms)),
                a)
            tot_loss = tf.add(tf.reduce_sum(batch_delta_e2), tf.exp(tf.multiply(a, tf.tanh(red_sum))), \
                              name = '1.LOSS-quad_exp_tanh_Delta_E2_div_Natom')
    
        elif loss_func == "quad_exp_tanh":
            a = tf.constant(5.0)
            red_sum = tf.div( tf.reduce_sum(batch_delta_e2), a)
            tot_loss = tf.add(tf.reduce_sum(batch_delta_e2), tf.exp(tf.multiply(a, tf.tanh(red_sum))), \
                              name = '1.LOSS-quad_exp_tanh_Delta_E2')

        with tf.name_scope('graphs_variable'):
            mean_delta_e = tf.reduce_mean(batch_delta_e)
            mean_delta_e2 = tf.reduce_mean(batch_delta_e2)
            energy_rmse = tf.sqrt((mean_delta_e2 - mean_delta_e**2),
                                  name='2.Energy_RMSE')

            mean_delta_e_atom = tf.reduce_mean(
                tf.div(batch_delta_e, batch_natoms))
            mean_delta_e2_atom = tf.reduce_mean(
                tf.div(batch_delta_e2, tf.square(batch_natoms)))
            energy_rmse_atom = tf.sqrt(
                (mean_delta_e2_atom - mean_delta_e_atom**2),
                name='3.Energy_RMSE_atom')

    tf.add_to_collection('losses', tot_loss)
    tf.add_to_collection('losses', energy_rmse_atom)
    tf.add_to_collection('losses', energy_rmse)

    return tot_loss, batch_delta_e


def loss_F(batch_species,
           batch_dEdG,
           batch_dgvect,
           batch_energies,
           forces_ref,
           batch_natoms,
           batch_size,
           Nspecies,
           gvect_size,
           floss_func="quad"):
    """ this is the cost function from forces terms

    Args:

    Returns:
        the loss value for the forces term

    """

    # I do this with a for loop, which will be quite slow...
    # but for now I don't see better options...
    all_forces = []
    ref_forces = []

    species_ind = tf.zeros([Nspecies], dtype=tf.int32)
    # Loop over molecules
    for m in range(batch_size):
        at = tf.where(tf.less(batch_species[m, :], Nspecies))
        if at is not []:
            nat = tf.size(at)
            ref_forces.append(forces_ref[m, 0:nat * 3])
            ngd = nat**2 * gvect_size * 3
            mol_dGdx = tf.reshape(batch_dgvect[m, 0:ngd],
                                  [nat, gvect_size, nat * 3])
            mol_F = tf.zeros([nat * 3])
            for s in range(Nspecies):
                atomic_index = tf.reshape(
                    tf.where(tf.equal(batch_species[m, :], s)), [-1])
                if atomic_index is not []:
                    nat_s = tf.size(atomic_index)
                    mol_dGdx_spec = tf.gather(mol_dGdx, atomic_index)
                    mol_dEdG_spec = tf.slice(batch_dEdG[s],
                                             [species_ind[s], 0], [nat_s, -1])
                    F_terms = -tf.einsum('ijk,ij->ik', mol_dGdx_spec,
                                         mol_dEdG_spec)
                    mol_F_spec = tf.reduce_sum(F_terms, axis=0)
                    mol_F = tf.add(mol_F, mol_F_spec)
                    species_ind = species_ind + nat_s * tf.one_hot(
                        s, depth=Nspecies, dtype=tf.int32)
            all_forces.append(mol_F)

    # Take differences and sum for loss
    lossF = tf.square(
        tf.concat(all_forces, axis=0) - tf.concat(ref_forces, axis=0))
    if floss_func == "quad":
        lossF = tf.reduce_sum(lossF, name='5.F_Loss')
    elif floss_func == "exp_quad":
        lossF = 0.5 * tf.exp(2.0 * tf.reduce_sum(lossF), name='5.F_Loss')
    elif floss_func == "quad_atom":
        lossF = tf.reduce_sum(
            tf.div(tf.reduce_sum(lossF, axis=-1), tf.square(batch_natoms)), name='5.F_Loss')
    elif floss_func == "exp_quad_atom":
        lossF = 0.5 * tf.exp(
            2.0 * tf.reduce_sum(tf.div(tf.reduce_sum(lossF, axis=-1), tf.square(batch_natoms))),
            name='5.F_Loss')

    tf.add_to_collection('losses', lossF)

    return lossF
