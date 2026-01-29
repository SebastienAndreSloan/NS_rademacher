# Instructions for training the PINN:
    The parameters for training and the data are defined at the top of the file. The two important ones are the d_Ns array and the N_rs array, as the program will loop over both of these arrays and train a PINN for each pair.

    The results are saved using the following two commands:
    torch.save(net_dict, f"weights_NS_tanh_dNs_{d_Ns}_Nrs_{N_rs}_i_{iterations}.pt")
    np.savez(f"result_NS_tanh_dNs_{d_Ns}_Nrs_{N_rs}_i_{iterations}",train=train_errors,test=test_errors)
    So there will be one pytorch file containing the trained weights, and one numpy file containing the training and testing errors.

# Instructions for plotting the training results:
    In Results.ipynb, The training v testing error plots are done for every N_rs, so that must be copied over from the training script. It must be rerun to see different d_Ns. Otherwise just replace the file name with the path to the numpy file containing the training and testing stats.

    To plot the outputs vs the true solution, change the file name to the path to the pytorch weight file, and make sure the net parameters are the same as the net having been trained, including the width of the middle layer below the definition of the neural net class.

# Instructions for plotting the Rademacher bounds:
    To be finished.
