#example: python3 plot-loss.py --dir ./checkpoints-7Sept2020-chairsSDHom

import argparse
import os
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dir",help="Directory of checkpoints which contains screenlog.0")
args = parser.parse_args()

if os.path.isfile(os.path.join(args.dir,"screenlog.0")):
    print("reading and parsing "+os.path.join(args.dir,"screenlog.0"))
    train_pattern = re.compile(r"""^\[\s+(?P<iteration>\d+),\s+(?P<learning_rate>\d*.?\d*)]\s+(?P<epe>\d*.?\d*),\s+(?P<onepx>\d*.?\d*),\s+(?P<threepx>\d*.?\d*),\s+(?P<fivepx>\d*.?\d*)""", re.VERBOSE)
    with open(os.path.join(args.dir,"screenlog.0")) as f:
        iteration = []
        learning_rate = []
        epe = []
        onepx = []
        threepx = []
        fivepx = []
        lines=f.readlines()
        for line in lines:
            match=train_pattern.match(line)
            if match:
                iteration.append(float(match.group("iteration")))
                learning_rate.append(float(match.group("learning_rate")))
                epe.append(float(match.group("epe")))
                onepx.append(float(match.group("onepx")))
                threepx.append(float(match.group("threepx")))
                fivepx.append(float(match.group("fivepx")))
        
    colors = ["b",'g','r','c','m','y','k']

    # THIS MAKES A GRID OF 1 ROW and len(stocks) COLUMN and figure size as (width, height) in inches.
    fig, axs = plt.subplots(2, 3, figsize=(30, 5))


    axs[0,0].plot(iteration,learning_rate,color=colors[0])
    axs[0,0].set_title("lr vs iter")
    axs[0,1].plot(iteration,epe,color=colors[1])
    axs[0,1].set_title("epe vs iter")
    axs[0,2].plot(iteration,onepx,color=colors[2])
    axs[0,2].set_title("1px vs iter")
    axs[1,0].plot(iteration,threepx,color=colors[3])
    axs[1,0].set_title("3px vs iter")
    axs[1,1].plot(iteration,fivepx,color=colors[4])
    axs[1,1].set_title("5px vs iter")

    #show plot
    plt.show()
    print(len(iteration),len(learning_rate),len(epe),len(onepx),len(threepx),len(fivepx))
else:
    print("Error: screenlog.0 file does not exist in "+str(args.dir))
    