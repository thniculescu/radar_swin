# %%
import numpy as np
import matplotlib.pyplot as plt 

# real_v = [
#     [-0.12, -5.3],
#     [0, -6.6],
#     [0, -6.6],
#     [-0.23, -8.05],
# ]
# real_v = np.array(real_v)


comp = np.load('/imec/other/dl4ms/nicule52/nuscenes-viz-pre/test_viteze/check_vit_poz_test.npy') 
# raw = np.load('/imec/other/dl4ms/nicule52/nuscenes-viz-pre/test_viteze/check_vit_poz_test.npy') # raw_man

pos = comp[:, :2]
comp = comp[:, 2:4]
# raw = raw[:, 2:4]


def project(v, u):
    return np.dot(v, u) / np.dot(u, u) * u

plt.figure(figsize=(12, 12))

#scatter points pos x,y
plt.scatter(pos[:, 0], pos[:, 1], label='pozitie')
plt.xlim(-40, 40)
plt.ylim(-40, 40)

#project raw, real_v vectors on line 0..x,y

# raw_proj = np.empty_like(raw)
comp_proj = np.empty_like(comp)
# real_proj = np.empty_like(real_v)




#make size of plot 16 x 16

for x in range(pos.shape[0]):
    # raw_proj[x] = project(raw[x], pos[x])
    comp_proj[x] = project(comp[x], pos[x])

    # plt.plot([pos[x, 0], 0], [pos[x, 1], 0], 'k--')
    # plt.arrow(pos[x, 0], pos[x, 1], real_v[x, 0], real_v[x, 1], head_width=0.5, head_length=0.5, fc='r', ec='r', label='viteza reala')
    # plt.arrow(pos[x, 0], pos[x, 1], raw[x, 0], raw[x, 1], head_width=0.5, head_length=0.5, fc='b', ec='b', label='viteza bruta')
    plt.arrow(pos[x, 0], pos[x, 1], comp[x, 0], comp[x, 1], head_width=0.5, head_length=0.5, fc='g', ec='g', label='compensated speed')

    # plt.arrow(pos[x, 0], pos[x, 1], real_proj[x, 0], real_proj[x, 1], head_width=0.5, head_length=0.5, fc='r', ec='r', label='viteza reala')
    # plt.arrow(pos[x, 0], pos[x, 1], raw_proj[x, 0], raw_proj[x, 1], head_width=0.5, head_length=0.5, fc='b', ec='b', label='viteza bruta')
    # plt.arrow(pos[x, 0], pos[x, 1], comp_proj[x, 0], comp_proj[x, 1], head_width=0.5, head_length=0.5, fc='g', ec='g', label='viteza calculata')

    #calculate slope of pos->comp
    slope = (comp[x, 1]) / (comp[x, 0])
    n = pos[x, 1] - slope * pos[x, 0]

    jj = np.linspace(0, pos[x, 0], 2)
    plt.plot(jj, slope * jj + n, '--')
    
plt.xlim(-2, 5) 
plt.ylim(-2, 2) 
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.grid()


