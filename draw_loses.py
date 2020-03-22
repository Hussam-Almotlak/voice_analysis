import ujson
import numpy as np
import matplotlib.pyplot as plt
import os

#importing the test loses from the json file
values_test = ujson.load(open("test_losses_vae_g_l.json", 'r'))
loss_test = [values_test[i]['loss'] for i in range(len(values_test))]
reconstruction_nll_test = [values_test[i]['reconstruction_nll'] for i in range(len(values_test))]
prior_nll_test = [values_test[i]['prior_nll'] for i in range(len(values_test))]
z_prediction_nll_test = [values_test[i]['z_prediction_nll'] for i in range(len(values_test))]
z_global_entropy_test = [values_test[i]['z_global_entropy'] for i in range(len(values_test))]
z_local_entropy_test = [values_test[i]['z_local_entropy'] for i in range(len(values_test))]

#importing the training losses from the jsom file
values_train = ujson.load(open("train_losses_vae_g_l.json", 'r'))
loss_train = [values_train[i]['loss'] for i in range(len(values_train))]
reconstruction_nll_train = [values_train[i]['reconstruction_nll'] for i in range(len(values_train))]
prior_nll_train = [values_train[i]['prior_nll'] for i in range(len(values_train))]
z_prediction_nll_train = [values_train[i]['z_prediction_nll'] for i in range(len(values_train))]
z_global_entropy_train = [values_train[i]['z_global_entropy'] for i in range(len(values_train))]
z_local_entropy_train = [values_train[i]['z_local_entropy'] for i in range(len(values_train))]

#creating a directory for the images
if not os.path.exists('Losses'):
    os.makedirs('Losses')

#plotting the Overall loss
plt.title('Overall Loss values over the iterations')
plt.plot(loss_test, label='Testing Overall Loss')
plt.plot(loss_train, label='Training Overall Loss')
plt.legend(loc='upper right')
plt.savefig('Losses/Overall_Loss.png')
plt.clf()

#plotting reconstruction loss
plt.title('Reconstruction Loss values over the iterations')
plt.plot(reconstruction_nll_test, label='Testing Reonstruction Loss')
plt.plot(reconstruction_nll_train, label='Training Reonstruction Loss')
plt.legend(loc='upper right')
plt.savefig('Losses/Reconstruction_Loss.png')
plt.clf()

#plotting the prior_nll
plt.title('Prior_Nll Loss values over the iterations')
plt.plot(prior_nll_test, label='Testing Prior_Nll')
plt.plot(prior_nll_train, label='Training Prior_Nll')
plt.legend(loc='upper right')
plt.savefig('Losses/Prior_Nll_Loss.png')
plt.clf()

#plotting the Z_Prediction_nll
plt.title('Z_Prediction_nll Loss values over the iterations')
plt.plot(z_prediction_nll_test, label='Testing Z_Prediction_nll')
plt.plot(z_prediction_nll_train, label='Training Z_Prediction_nll')
plt.legend(loc='upper right')
plt.savefig('Losses/Z_Prediction_Nll_Loss.png')
plt.clf()

#plotting the z_global_entropy
plt.title('Z_Global_Entropy Loss values over the iterations')
plt.plot(z_global_entropy_test, label='Testing Z_Global_Entropy')
plt.plot(z_global_entropy_train, label='Training Z_Global_Entropy')
plt.legend(loc='upper right')
plt.savefig('Losses/Z_Global_Entopy_Loss.png')
plt.clf()

#plotting the z_local_entropy
plt.title('Z_Local_Entropy Loss values over the iterations')
plt.plot(z_local_entropy_test, label='Testing Z_Local_Entropy')
plt.plot(z_local_entropy_train, label='Training Z_Local_Entropy')
plt.legend(loc='upper right')
plt.savefig('Losses/Z_Local_Entropy_Loss.png')
plt.clf()