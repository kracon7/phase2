import numpy as np
from sympy import *


def func(target):
	e1 = np.array([-0.0269, 0.7183, 0.6952])
	vx = np.array([-0.8163, -0.4171, 0.3995])
	c = np.cross(e1, vx)
	d = e1 @ vx

	origin = np.array([32.4282, -81.5175, -147.657])
	V = target - origin

	theta = symbols('theta')
	e1 = Eq((vx @ V) * cos(theta) + (c @ V) * sin(theta) + (d * e1 @ V) * (1-cos(theta)), 59.7)
	result = solve(e1, theta)

	print(result)


# test target position at [0, 0]
target = 100*np.array([0.420562588779785,0.0447626641885372,0.906158602460734]) + np.array([
	-15.7957793261782, -107.336773592419, -124.302432183635])

func(target)

# test target position at [90, 0]
target = 100*np.array([0.0432403231797374, -0.0440565772727703, 0.998092827571825]) + np.array([
	-26.6342699711842,-84.5642703841322,-143.753255913368])

func(target)
