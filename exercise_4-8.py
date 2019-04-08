import numpy as np

# The root system of O]T
r1 = np.array([1,0,0])
r2 = np.array([0,1,0])
r3 = np.array([0,0,1])

# The symmetries of r1, r2, r3
s1 = np.array([[0,1,0],[1,0,0],[0,0,1]])
s2 = np.array([[1,0,0],[0,0,1],[0,1,0]])
s3 = np.array([[0,-1,0],[-1,0,0],[0,0,1]])

# The change of basis matrix from canonical to r1, r2, r3
T = np.array([[1,0,-1],[-1,1,-1],[0,-1,0]])
Tinv = np.linalg.inv(T)

s1prime = np.matmul(Tinv, s1)
s1prime = np.matmul(s1prime, T)

s2prime = np.matmul(Tinv, s2)
s2prime = np.matmul(s2prime, T)

s3prime = np.matmul(Tinv, s3)
s3prime = np.matmul(s3prime, T)


# The t-base for O]T
generating_reflections = [s1prime, s2prime, s3prime]
group_elements = []

group_elements.extend([s1prime, s2prime, s3prime])

print("The elements of O]T are")

print(s1prime)
print(s2prime)
print(s3prime)

print(len(group_elements))

for i in range(3):
	for j in range(3):
		temp = np.matmul(generating_reflections[i],generating_reflections[j])
		for index in range(len(group_elements)):
			if np.array_equal(group_elements[index], temp):
				break
			if index == len(group_elements)-1:
				group_elements.append(temp)
				print(temp)

print(len(group_elements))

for i in range(3):
	for j in range(3):
		for k in range(3):
			temp = np.matmul(generating_reflections[i], generating_reflections[j])
			temp = np.matmul(temp, generating_reflections[k])
			length = len(group_elements)
			for index in range(length):
				if np.array_equal(group_elements[index], temp):
					break
				if index == length - 1:
					group_elements.append(temp)
					print(temp)

print(len(group_elements))

for i in range(3):
	for j in range(3):
		for k in range(3):
			for l in range(3):
				temp = np.matmul(generating_reflections[i], generating_reflections[j])
				temp = np.matmul(temp, generating_reflections[k])
				temp = np.matmul(temp, generating_reflections[l])
				length = len(group_elements)
				for index in range(length):
					if np.array_equal(group_elements[index], temp):
						break
					if index == length - 1:
						group_elements.append(temp)
						print(temp)

print(len(group_elements))

for i in range(3):
	for j in range(3):
		for k in range(3):
			for l in range(3):
				for m in range(3):					
					temp = np.matmul(generating_reflections[i], generating_reflections[j])
					temp = np.matmul(temp, generating_reflections[k])
					temp = np.matmul(temp, generating_reflections[l])
					temp = np.matmul(temp, generating_reflections[m])
					length = len(group_elements)
					for index in range(length):
						if np.array_equal(group_elements[index], temp):
							break
						if index == length - 1:
							group_elements.append(temp)
							print(temp)

print(len(group_elements))

for i in range(3):
	for j in range(3):
		for k in range(3):
			for l in range(3):
				for m in range(3):	
					for n in range(3):				
						temp = np.matmul(generating_reflections[i], generating_reflections[j])
						temp = np.matmul(temp, generating_reflections[k])
						temp = np.matmul(temp, generating_reflections[l])
						temp = np.matmul(temp, generating_reflections[m])
						temp = np.matmul(temp, generating_reflections[n])
						length = len(group_elements)
						for index in range(length):
							if np.array_equal(group_elements[index], temp):
								break
							if index == length - 1:
								group_elements.append(temp)
								print(temp)

print(len(group_elements))

root_system = [r1, r2, r3]

print("The root system for O]T is:")

print(r1, r2, r3)

for element in group_elements:
	temp = np.matmul(element, r1)
	length = len(root_system)
	for index in range(length):
		if np.array_equal(root_system[index], temp):
			break
		if index == length - 1:
			root_system.append(temp)
			print(temp)

for element in group_elements:
	temp = np.matmul(element, r2)
	length = len(root_system)
	for index in range(length):
		if np.array_equal(root_system[index], temp):
			break
		if index == length - 1:
			root_system.append(temp)
			print(temp)

for element in group_elements:
	temp = np.matmul(element, r3)
	length = len(root_system)
	for index in range(length):
		if np.array_equal(root_system[index], temp):
			break
		if index == length - 1:
			root_system.append(temp)
			print(temp)

print(len(root_system))



