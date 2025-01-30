import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.animation import FuncAnimation



def animate_motion_with_edges(motion, edges):
    """
    Animates the 3D motion sequence with edges between joints.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Motion Animation with Edges")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    
    # Initialize plot elements for joints and edges
    joint_markers = [ax.plot([], [], [], 'o', lw=2, color='blue')[0] for _ in range(motion.shape[1])]
    edge_lines = [ax.plot([], [], [], '-', lw=1, color='red')[0] for _ in edges]
    
    # Set plot limits based on motion data
    max_range = np.max(np.ptp(motion, axis=(0, 1)))  # Range of motion
    center = np.mean(motion, axis=(0, 1))  # Center of motion
    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
    
    def update(frame_idx):
        """
        Update function for animation.
        """
        # Update joint markers
        for joint_idx, marker in enumerate(joint_markers):
            joint_x = motion[frame_idx, joint_idx, 0]
            joint_y = motion[frame_idx, joint_idx, 1]
            joint_z = motion[frame_idx, joint_idx, 2]
            marker.set_data([joint_x], [joint_y])  # Update x, y
            marker.set_3d_properties([joint_z])   # Update z
        
        # Update edges
        for edge_idx, (start, end) in enumerate(edges):
            edge_x = [motion[frame_idx, start, 0], motion[frame_idx, end, 0]]
            edge_y = [motion[frame_idx, start, 1], motion[frame_idx, end, 1]]
            edge_z = [motion[frame_idx, start, 2], motion[frame_idx, end, 2]]
            edge_lines[edge_idx].set_data(edge_x, edge_y)  # Update x, y
            edge_lines[edge_idx].set_3d_properties(edge_z)  # Update z
        
        return joint_markers + edge_lines

    ani = FuncAnimation(fig, update, frames=motion.shape[0], interval=50, blit=True)
    plt.show()

def animate_motion(motion):
    """
    Animates the 3D motion sequence.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Motion Animation")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    
    # Initialize plot elements for each joint
    lines = [ax.plot([], [], [], 'o-', lw=2)[0] for _ in range(motion.shape[1])]
    
    # Set plot limits based on motion data
    max_range = np.max(np.ptp(motion, axis=(0, 1)))  # Range of motion
    center = np.mean(motion, axis=(0, 1))  # Center of motion
    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
    
    def update(frame_idx):
        """
        Update function for animation.
        """
        for joint_idx, line in enumerate(lines):
            joint_x = motion[frame_idx, joint_idx, 0]
            joint_y = motion[frame_idx, joint_idx, 1]
            joint_z = motion[frame_idx, joint_idx, 2]
            line.set_data([joint_x], [joint_y])  # Update x, y
            line.set_3d_properties([joint_z])   # Update z
        return lines

    ani = FuncAnimation(fig, update, frames=motion.shape[0], interval=50, blit=True)
    plt.show()

def plot_motion(motion):
    """
    Plots the 3D motion sequence.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Motion Visualization")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    
    for frame in motion:
        xs, ys, zs = frame[:, 0], frame[:, 1], frame[:, 2]
        ax.plot(xs, ys, zs, 'o-', alpha=0.7)
    plt.show()

def load_text(file_path):
    """
    Loads and prints the text description.
    """
    with open(file_path, 'r') as f:
        text = f.read().strip()
    return text

# Paths to folders
motion_folder = "../KIT-ML-20250125T102306Z-001/KIT-ML/new_joints/new_joints"  # Replace with the path to the 'new_joints' folder
text_folder = "../KIT-ML-20250125T102306Z-001/KIT-ML/texts/texts"         # Replace with the path to the 'texts' folder

# Example: Choose a specific motion
motion_file = os.path.join(motion_folder, "M02574.npy")  # Replace with the motion file name
text_file = os.path.join(text_folder, "M02574.txt")      # Replace with the text file name

# Load motion and text
motion = np.load(motion_file)  # Shape: (num_frames, num_joints, 3)
description = load_text(text_file)


edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),        # Spine
    (3, 6), (6, 7), (7, 8), (8, 9),               # Left Arm
    (3, 10), (10, 11), (11, 12), (12, 13),        # Right Arm
    (0, 14), (14, 15), (15, 16),                  # Left Leg
    (0, 17), (17, 18), (18, 19)                   # Right Leg
]
# Display motion and text
print("Motion Description:", description)
animate_motion_with_edges(motion,edges)