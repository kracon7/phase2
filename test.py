import numpy as np
from sympy import *


def func(target):
	# rotation axis e1
	e1 = np.array([0.0269, -0.7183, -0.6952])
	# reverse of x_axis vx
	vx = np.array([-0.8163, -0.4171, 0.3995])
	vz = np.array([0.420562588779785, 0.0447626641885372, 0.906158602460734])

	theta = symbols('theta')
	L = 59.7

	origin = np.array([32.4282, -81.5175, -147.657])
	V = target - origin

	expr_1 = vx * cos(theta) + np.cross(e1, vx) * sin(theta) + e1 @ vx * e1 * (1-cos(theta))
	eq1 = Eq(expr_1 @ V, L)
	result = solve(eq1, theta)

	if len(result) > 0:
		for th1 in result:
			angle_1 = np.rad2deg(float(th1))
			
			# if valid angle for rotation 1 is found
			if angle_1 > -1e-2 and angle_1 < 60:
				theta_1 = float(th1)

				# keep solving for theta_2
				rotated_x = np.zeros(3)
				for i in range(3):
					rotated_x[i] = expr_1[i].subs(theta, theta_1)

				origin_2 = origin + L * rotated_x

				vector = target - origin_2
				vector = vector / np.linalg.norm(vector)
				rotated_z = vz*np.cos(theta_1) + np.cross(e1, vz)*np.sin(theta_1) \
							+ e1@vz * e1 * (1-np.cos(theta_1))

				angle_2 = np.rad2deg(np.arccos(rotated_z @ vector))
				if angle_2 >= 0 and angle_2 <= 20:	
					print([angle_1, angle_2])
					return [angle_1, angle_2]

	return None


# test target position at [0, 0]
target = 100*np.array([0.420562588779785,0.0447626641885372,0.906158602460734]) + np.array([
	-15.7957793261782, -107.336773592419, -124.302432183635])

func(target)

# test target position at [90, 0]
target = 100*np.array([0.0432403231797374, -0.0440565772727703, 0.998092827571825]) + np.array([
	-26.6342699711842,-84.5642703841322,-143.753255913368])

func(target)
