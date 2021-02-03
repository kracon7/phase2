import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mujoco_py import load_model_from_path, MjSim, MjViewer

# plt.ion()
# fig, axes = plt.subplots(1,2)

def weed_in_range(laser_xpos, weed_xpos):
	laser_y = laser_xpos[1]
	for i, pos in enumerate(weed_xpos):
		y = pos[1]
		if y - laser_y < 0.01 and y - laser_y > -0.03:
			return True, i
	return False, None

def turn_on_laser(sim, beam_geomid):
	sim.model.geom_rgba[beam_geomid,:] = np.array([1., 0., 0., 1.])
	sim.forward()

def turn_off_laser(sim, beam_geomid):
	sim.model.geom_rgba[beam_geomid,:] = np.array([1., 1., 1., .0])
	sim.forward()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

xml_path =os.path.join(ROOT_DIR, 'rover_demo.xml')
# xml_path = './rover_demo.xml'
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

rover_body_id = model._body_name2id['rover']
rover_y_jid = 1
laser_body_id = model._body_name2id['laser_rotor']
laser_slide_jid = 7
laser_roter_jid = 8
laser_geomid = model._geom_name2id['laser']
beam_geomid = model._geom_name2id['beam']
weed_names = ['weed_0', 'weed_1', 'weed_2', 'weed_3']
weed_geomid = [model._geom_name2id[name] for name in weed_names]
weed_xpos = [sim.data.geom_xpos[idx] for idx in weed_geomid]
weed_xpos = np.stack(weed_xpos) - np.array([0., 0., 0.3])


while True:
	sim.reset()
	sim.data.qpos[rover_y_jid] = 2
	sim.forward()
	for i in range(200):
		sim.step()
		viewer.render()

	turn_off_laser(sim, beam_geomid)

	step_size = 1e-3
	tracking_begins = False
	tracking_time = 0
	tracking_length = 800
	cd = 0
	for i in range(13000):
		# move rover forward one step_size
		sim.data.qpos[rover_y_jid] = -i * step_size
		sim.forward()

		if cd == 0:
			laser_xpos = sim.data.geom_xpos[laser_geomid]
			in_range, idx = weed_in_range(laser_xpos, weed_xpos)
			if in_range:
				tracking_begins = True
				weed_id = idx

			if tracking_begins:
				# compute angle
				dx = np.abs(laser_xpos[0] - weed_xpos[weed_id, 0])
				dz = np.abs(laser_xpos[2] - weed_xpos[weed_id, 2])
				angle = np.arctan2(dz, dx) - np.pi/4

				# if in shooting range
				if tracking_time < tracking_length:
					if np.abs(weed_xpos[weed_id, 1] - laser_xpos[1]) < 0.02:
						turn_on_laser(sim, beam_geomid)
					sim.forward()
					# car center + 1.6 + slide_joint = weed_y
					sim.data.qpos[laser_slide_jid] = weed_xpos[weed_id, 1] - sim.data.qpos[1] - 2.45
					sim.data.qpos[laser_roter_jid] = angle
					tracking_time += 1
				else:
					tracking_time = 0
					tracking_begins = False
					turn_off_laser(sim, beam_geomid)
					cd = 1
			else:
				if sim.data.qpos[laser_slide_jid] > -0.35:
					sim.data.qpos[laser_slide_jid] -= step_size
				elif sim.data.qpos[laser_slide_jid] < -0.25:
					sim.data.qpos[laser_slide_jid] += step_size
				turn_off_laser(sim, beam_geomid)
		elif cd > 0 and cd < 200:
			cd += 1
		else:
			cd = 0

		sim.forward()
		viewer.render()