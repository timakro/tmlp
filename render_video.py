#!/usr/bin/env python3
import gzip
import os
import json

import numpy as np
from matplotlib import pyplot as plt, animation, colors, patches


SESSIONS_DIR = '/home/tim/lang/python/tensorflow/tmlp/sessions'
SESSION_ID = '5cff88ae-timakro.de-1-0-fy5itVte'

LABEL_DTYPE = np.dtype([
    ('targetx', np.int32),
    ('targety', np.int32),
    ('direction', np.int8),
    ('weapon', np.int8),
    ('jump', np.bool),
    ('fire', np.bool),
    ('hook', np.bool),
])

WEAPON_NAMES = {
    0: 'Hammer',
    1: 'Gun',
    2: 'Shotgun',
    3: 'Grenade',
    4: 'Rifle',
    5: 'Ninja'
}

COLORMAP = [
    '#ffffff', # Void
    '#989898', # Solid tile
    '#c5656c', # Death tile
    '#5a5a5a', # Nohook tile
    '#b1179e', # Health pickup
    '#6aaed7', # Armor pickup
    '#6a3c00', # Shotgun pickup
    '#a4000a', # Grenade pickup
    '#101a74', # Rifle pickup
    '#171717', # Ninja pickup
    '#64ed63', # Gun projectile
    '#6a3c00', # Shotgun projectile
    '#a4000a', # Grenade projectile
    '#101a74', # Laser beam
    '#dbc311', # Player with hammer
    '#64ed63', # Player with gun
    '#6a3c00', # Player with shotgun
    '#a4000a', # Player with grenade
    '#101a74', # Player with rifle
    '#171717', # Player with ninja
    '#171717', # Player hook
    '#171717'  # HUD
]

session_dir= os.path.join(SESSIONS_DIR, SESSION_ID)

with open(os.path.join(session_dir, 'meta.json')) as metafile:
    meta = json.load(metafile)

datafile = gzip.GzipFile(os.path.join(session_dir, 'data.gz'))
binary_data = datafile.read()
data = np.frombuffer(binary_data, dtype=np.uint8)
data = data.reshape((-1, 50, 90))

labelsfile = gzip.GzipFile(os.path.join(session_dir, 'labels.gz'))
binary_labels = labelsfile.read()
labels = np.frombuffer(binary_labels, dtype=LABEL_DTYPE)

print(meta)
print(data.shape)
print(labels.shape)

cmap = colors.ListedColormap(COLORMAP)
norm = colors.BoundaryNorm([k-0.5 for k in range(cmap.N+1)], cmap.N)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.6), gridspec_kw={'width_ratios': [2, 1]})
ax1.set_title('Player vision')
ax2.set_title('Player input')
ax2.axis([-500, 500, 500, -500])
ax2.set_aspect('equal')
ax2.grid(True)
ax2.set_xticks([-500, 0, 500])
ax2.set_yticks([-500, 0, 500])


images = []
for frame, label in zip(data, labels):
    scene = []
    scene.append(ax1.imshow(frame, cmap=cmap, norm=norm))
    scene.append(ax2.plot(label['targetx'], label['targety'], 'k+')[0])
    if label['direction'] in (1, -1):
        scene.append(ax2.arrow(0, 0, 100*label['direction'], 0, width=25))
    scene.append(ax2.text(-400, -400, WEAPON_NAMES[label['weapon']], color=COLORMAP[14+label['weapon']], weight='bold'))
    if label['jump']:
        scene.append(ax2.arrow(0, 0, 0, -100, width=25))
    if label['fire']:
        scene.append(ax2.add_patch(patches.Arc((label['targetx'], label['targety']), 100, 150, theta1=90, theta2=270)))
    if label['hook']:
        scene.append(ax2.add_patch(patches.Arc((label['targetx'], label['targety']), 100, 150, theta1=270, theta2=90)))
    images.append(scene)
ani = animation.ArtistAnimation(fig, images, interval=40, blit=True, repeat=False)
plt.show()

#ani.save('movie.mp4', dpi=300)
