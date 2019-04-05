"""
Author: Vinicius C. Costa
Date: Mar 6, 2019.

Finite Reflection Groups

This is a program to show that there are only three standalone
finite groups in three dimensions that are generated
by reflections they contain: 
The O*, or octahedral group together with -1, 
The I*, or icosahedral group together with -1, and
The O]T, the set of all symmetries of the tetrahedron.
"""

import numpy as np

cube = list(range(8))
icosahedron = list(range(12))

reflections_O_star = np.array([[0,6,5,3,4,2,1,7],
							   [7,1,2,4,3,5,6,0],
							   [7,4,5,6,3,0,1,2],
							   [2,1,0,3,6,5,4,7],
							   [0,3,2,1,4,7,6,5],
							   [5,4,2,3,1,0,6,7],
							   [6,7,4,5,2,3,0,1],
							   [1,0,3,2,5,4,7,6],
							   [3,2,1,0,7,6,5,4]])

reflections_I_star = np.array([[8,7,6,4,5,3,2,1,12,10,11,9],
							   [9,5,7,4,2,6,3,11,1,10,8,12],
							   [4,9,8,1,5,6,10,3,2,7,11,12],
							   [1,2,12,11,10,9,7,8,6,5,4,3],
							   [1,11,3,12,8,10,7,5,9,6,2,4],
							   [10,2,3,7,12,11,4,8,9,1,6,5],
							   [6,10,3,8,5,1,12,4,9,2,11,7],
							   [11,6,3,4,7,2,5,12,9,10,1,8],
							   [1,4,11,2,9,6,7,10,5,8,3,12],
							   [1,12,4,3,5,8,7,6,10,9,11,2],
							   [12,2,5,4,3,7,6,8,11,10,9,1],
							   [5,2,10,9,1,6,11,8,4,3,7,12],
							   [3,2,1,6,5,4,9,8,7,12,11,10],
							   [2,1,3,5,4,6,8,7,9,11,10,12],
							   [1,3,2,4,6,5,7,9,8,10,12,11]])

reflections_I_star = reflections_I_star - 1

reflections_O_brack_T = np.array([[2,1,3,4,6,5,7,8],
								  [3,2,1,4,7,6,5,8],
								  [4,2,3,1,8,6,7,5],
								  [1,3,2,4,5,7,6,8],
								  [1,4,3,2,5,8,7,6],
								  [1,2,4,3,5,6,8,7]])

reflections_O_brack_T = reflections_O_brack_T - 1


def transform(symmetry, arr):
	answer = []
	for i in range(len(symmetry)): 
		j = symmetry[i]
		answer.insert(i, arr[j]) 
	return answer

def generate(reflections, polyhedron, depth):
	image_of_reflections = set()
	reflections_length = len(reflections)

	image_of_reflections.add(tuple(polyhedron))
	# print(tuple(polyhedron))

	print(len(image_of_reflections))

	if depth > 0:
		for i in range(reflections_length):
			symmetry = reflections[i]
			# if tuple(symmetry) not in image_of_reflections:
			# 	print(tuple(symmetry))
			image_of_reflections.add(tuple(symmetry))

	print(len(image_of_reflections))

	if depth > 1:
		for i in range(reflections_length):
			for j in range(reflections_length):
				if i!= j:
					symmetry = transform(reflections[j], reflections[i])
					# if tuple(symmetry) not in image_of_reflections:
					# 	print(tuple(symmetry))
					image_of_reflections.add(tuple(symmetry))
					
	print(len(image_of_reflections))

	if depth > 2:
		for i in range(reflections_length):
			for j in range(reflections_length):
				if i != j:
					for k in range(reflections_length):
						if j != k:
							symmetry = transform(reflections[j], reflections[i])
							symmetry = transform(reflections[k], symmetry)
							# if tuple(symmetry) not in image_of_reflections:
							# 	print(tuple(symmetry))
							image_of_reflections.add(tuple(symmetry))

	print(len(image_of_reflections))

	if depth > 3:
		for i in range(reflections_length):
			for j in range(reflections_length):
				if i != j:
					for k in range(reflections_length):
						if j != k:
							for l in range(reflections_length):
								if k != l:
									symmetry = transform(reflections[j], reflections[i])
									symmetry = transform(reflections[k], symmetry)
									symmetry = transform(reflections[l], symmetry)
									# if tuple(symmetry) not in image_of_reflections:
									# 	print(tuple(symmetry))
									image_of_reflections.add(tuple(symmetry))

	# for element in image_of_reflections:
	# 	print(element)

	print(len(image_of_reflections))

generate(reflections_O_star, cube, 4)

generate(reflections_O_brack_T, cube, 3)

generate(reflections_I_star, icosahedron, 3)




		
