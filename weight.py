import sys
import os
import re
import tensorflow as tf

def save_weights(saver, sess, epoch_i):
    path = "./params/"
    if not os.path.exists(path):
        os.makedirs(path)
    saver.save(sess, path + "epoch" + str(epoch_i))


def load_weights_with_confirm(saver, sess):
    path = "./params"
    if not os.path.exists(path):
        os.makedirs(path)

    params_files_name = [obj for obj in os.listdir(path) if os.path.isfile(path + "/" + obj)]
    ckpt_files = [file_name for file_name in params_files_name if ".data" in file_name]

    if ckpt_files:
        params_files = [[int(re.findall("\d+", file_name)[0]), file_name] for file_name in ckpt_files]
        params_files.sort()

        epochs = [i[0] for i in params_files]

        print("The parameter file  was found.")
        s = input("Do you wanna use it? (y/n) : ")
        if s == "y" or s == "Y":
            print("What epochs number do you want to start with?")
            for i in params_files:
                print("epoch ", i[0])
            epoch_input = int(input("Please enter the number : "))

            if epoch_input in epochs:
                params_file_name = params_files[epochs.index(epoch_input)][1]
                saver.restore(sess, path + "/epoch" + str(epoch_input))
                return epoch_input + 1
            else:
                print("The number you entered is invalid.")

    return 0