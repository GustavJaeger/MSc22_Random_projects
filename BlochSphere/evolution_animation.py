import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm


initial_state = np.array([[0],
                          [1]])
#pauli matrices and identity
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])


#function that calculates the expectation value
def expectation_value(state, operator):
    e = state.T @ operator @ state
    return float(e.real)


s_vector = np.array([[expectation_value(initial_state, X)],
                    [expectation_value(initial_state, Y)],
                    [expectation_value(initial_state, Z)]])

u = 1/np.sqrt(2) * np.array([[1, 1, 0]])

M = np.array([[0, -u[0][2], u[0][1]], 
             [u[0][2], 0, -u[0][0]],
             [-u[0][1], u[0][0], 0]])


def evolution_s(time, s_vector, M):
    s = expm(time * M) @ s_vector
    return s

def plot_arrow(vector, name, color_, brightness):
    ax.quiver(0, 0, 0, vector[0][0], vector[0][1], vector[0][2],
              arrow_length_ratio=0.05, color=color_, label=name, alpha=brightness)

def sphere_animation(N, tf, time_step, s_vector, M, L):
    global Q
    if N<=time_step:
        time = N/time_step * tf
        l_vector = evolution_s(time, s_vector, M)
        l_vector = l_vector.T      
        Q.remove()
        L.remove()
        ax.scatter(l_vector[0][0], l_vector[0][1], l_vector[0][2], c='blue', s=0.4, alpha=0.5)
        Q = ax.quiver(0, 0, 0, l_vector[0][0], l_vector[0][1], l_vector[0][2],
                      arrow_length_ratio=0.05, color='blue', label=r'$\mathbf{s}(\omega t='+str(round(time/np.pi, 4))+'\pi)$')
        L = ax.legend()
        return Q,
    else:
        if N == time_step +1: 
            Q.remove()
            L.remove()
        ax.view_init(30, 20 + N - time_step)
        ax.legend()


#sphere plot
phi, theta = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j] #equivalent to meshgrid but more compact syntax
x = np.cos(phi) * np.sin(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(theta)

# Create the figure
fig = plt.figure()

# Add an axes
ax = fig.gca(projection='3d')

# plot the surface
surf = ax.plot_wireframe(x, y, z, alpha=0.3, color='silver', zorder = 3, linewidths=1.0)

# Remove gray panes and axis grid
ax.xaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.fill = False
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.fill = False
ax.zaxis.pane.set_edgecolor('white')
ax.grid(False)


# Remove axis ticks
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])
ax.w_xaxis.line.set_lw(0.)
ax.set_xticks([])
ax.w_yaxis.line.set_lw(0.)
ax.set_yticks([])
ax.set_position([0,0, 1, 1])

# fake axis
ax.text(0, 0, 1.11, r'$z$', c='black')

ax.text(1.2, 0, 0, r'$x$', c='black')

ax.text(0, 1.11, 0, r'$y$', c='black')

p0 = [1.1, 0, 0]
p1 = [0, 1.1, 0]
p2 = [0, 0, 1.1]

origin = [0,0,0]
xx, yy, zz = zip(origin,origin,origin) 
U, V, W = zip(p0,p1,p2)
ax.quiver(xx,yy,zz,U,V,W,arrow_length_ratio=0.05, color='black', alpha=0.8)
#plot of vector u        
plot_arrow(u, r'$\mathbf{u}$', 'orange', 1)
plot_arrow(-u, '','orange', 0.5)
ax.quiver(0, 0, 0, s_vector[0][0], s_vector[1][0], s_vector[2][0], label = r'$\mathbf{s}(\omega t=0)$',
              arrow_length_ratio=0.05, color='green')

Q = ax.quiver(0, 0, 0, s_vector[0][0], s_vector[1][0], s_vector[2][0], pivot = 'tail',
              arrow_length_ratio=0.05, color='blue')
L = ax.legend()

ax.view_init(30, 20)    

time_steps = 250
iterations_max = time_steps + 360
tf = 4*np.pi

ani = animation.FuncAnimation(
    fig, sphere_animation, iterations_max, fargs=(tf, time_steps, s_vector, M, L), interval=50, blit=True)
data = {'title': 'BS_evolution', 'author':'Arnau Lira Solanilla'}
ani.save('animation_BS.mp4', dpi=700, metadata = data)
#f = r"c://Users/Arnau/Desktop/animation.mp4" 
#writermp4 = animation.FFMpegWriter(fps=60) 
#ani.save(f, writer=writermp4)
plt.show()
