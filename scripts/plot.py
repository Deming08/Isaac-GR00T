import numpy as np
import matplotlib.pyplot as plt

joint_name = ["Right_Shoulder_Pitch", 
              "Right_Shoulder_Roll", 
              "Right_Shoulder_Yaw", 
              "Right_Elbow", 
              "Right_Wrist_Roll", 
              "Right_Wrist_Pitch", 
              "Right_Wrist_Yaw", 
              ]

path = "/home/asus/Isaac-GR00T/output/action_valid/"
action_horizon = 16
num = 1

if __name__ == "__main__":

    pred_action_across_time = np.load(f"{path}gr00t_pred_action_seqs_{num}.npy")
    steps = len(pred_action_across_time)
    action_dim = pred_action_across_time.shape[1]

    joints_group1 = range(0, 3)
    joints_group2 = range(3, action_dim)


    fig1, axes1 = plt.subplots(nrows=len(joints_group1), ncols=1, figsize=(8, 4 * len(joints_group1)))
    if len(joints_group1) == 1: 
        axes1 = [axes1]

    for idx, joint in enumerate(joints_group1):
        ax = axes1[idx]
        ax.plot(pred_action_across_time[:, joint], label="Pred Action", color="green")
        ax.plot(range(steps), pred_action_across_time[:, joint], 'o', color="red", markersize=1, label="Data Point" if idx == 0 else "")

        for j in range(0, steps, action_horizon):
            ax.axvline(x=j, color="black", linestyle="--", linewidth=0.5, label="Inference Point" if j == 0 else "")
        
        ax.set_title(f"Joint {joint_name[joint]}")

    fig1.suptitle("Joint Actions (Group 1: Joints 0-2)", fontsize=16, color="blue")
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) 


    fig2, axes2 = plt.subplots(nrows=len(joints_group2), ncols=1, figsize=(8, 4 * len(joints_group2)))
    if len(joints_group2) == 1: 
        axes2 = [axes2]

    for idx, joint in enumerate(joints_group2):
        ax = axes2[idx]
        ax.plot(pred_action_across_time[:, joint], label="Pred Action", color="green")
        ax.plot(range(steps), pred_action_across_time[:, joint], 'o', color="red", markersize=1, label="Data Point" if idx == 0 else "")

        for j in range(0, steps, action_horizon):
            ax.axvline(x=j, color="black", linestyle="--", linewidth=0.5, label="Inference Point" if j == 0 else "")
            
        ax.set_title(f"Joint {joint_name[joint]}")

    fig2.suptitle("Joint Actions (Group 2: Joints 3-6)", fontsize=16, color="blue")
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) 

    

    plt.show()