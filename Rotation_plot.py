import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import all_functions as af

n = 6001//20 +1
t_lst = np.linspace(0,6000,n)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame_idx):
    ax.cla()

    t = t_lst[frame_idx]
    sigma, omega = af.Attitude_Simulation(t)
    dcm = af.MRP_to_DCM(sigma)

    b1 = dcm[0, :]
    b2 = dcm[1, :]
    b3 = dcm[2, :]

    origin = [0, 0, 0]

    inertial_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    u_n, v_n, w_n = zip(*inertial_axes)
    ax.quiver(0, 0, 0, u_n, v_n, w_n, color='black', linestyle='--', alpha=0.3)
    u_b, v_b, w_b = zip(b1, b2, b3)
    ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], u_b, v_b, w_b, color=['r', 'g', 'b'], linewidth=3)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('N1'); ax.set_ylabel('N2'); ax.set_zlabel('N3')
    ax.set_title(f'Attitude Simulation | Time: {t:.2f}s')
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', linestyle='--'),
                    Line2D([0], [0], color='r', lw=2)]
    ax.legend(custom_lines, ['Inertial', 'Body'], loc='upper left')

ani = FuncAnimation(fig, update, frames=len(t_lst), interval=50)

# 4. Save the Video
# Note: You need ffmpeg installed on your computer for this to work!
try:
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("attitude_video.mp4", writer=writer)
    print("Video saved successfully as attitude_video.mp4")
except Exception as e:
    print(f"Error saving video: {e}")
    print("Showing plot instead...")
    plt.show()

ani.save("attitude_animation.gif", writer='pillow')