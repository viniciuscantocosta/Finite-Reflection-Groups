import numpy as np

phi = (1+5**(0.5))/2
# print("{}\n".format(phi))
alpha = phi/2 
beta = 1/(2*phi)
precision = 3

# The symmetries of r1, r2, r3
s1prime = np.array([[-beta,-0.5,alpha],[-0.5,alpha,beta],[alpha,beta,0.5]])
s1prime = np.round_(s1prime, decimals = precision)
s2prime = np.array([[-beta,0.5,alpha],[0.5,alpha,-beta],[alpha,-beta,0.5]])
s2prime = np.round_(s2prime, decimals = precision)
s3prime = np.array([[0.5,alpha,-beta],[alpha,-beta,0.5],[-beta,0.5,alpha]])
s3prime = np.round_(s3prime, decimals = precision)

# # The change of basis matrix from canonical to r1, r2, r3
# T = np.array([[1,0,-1],[-1,1,-1],[0,-1,0]])
# Tinv = np.linalg.inv(T)

# s1prime = np.matmul(Tinv, s1)
# s1prime = np.matmul(s1prime, T)

# s2prime = np.matmul(Tinv, s2)
# s2prime = np.matmul(s2prime, T)

# s3prime = np.matmul(Tinv, s3)
# s3prime = np.matmul(s3prime, T)


# The t-base for O]T
generating_reflections = [s1prime, s2prime, s3prime]
group_elements = []

group_elements.extend([s1prime, s2prime, s3prime])

print("The elements of I_3 are")

print(s1prime)
print(s2prime)
print(s3prime)

print(len(group_elements))
for i in range(3):
	for j in range(3):
		temp = np.matmul(generating_reflections[i],generating_reflections[j])
		temp = np.round_(temp, decimals = precision)
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
			if j == k:
				continue
			temp = np.matmul(generating_reflections[i], generating_reflections[j])
			temp = np.matmul(temp, generating_reflections[k])
			temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				temp = np.matmul(generating_reflections[i], generating_reflections[j])
				temp = np.matmul(temp, generating_reflections[k])
				temp = np.matmul(temp, generating_reflections[l])
				temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue					
					temp = np.matmul(generating_reflections[i], generating_reflections[j])
					temp = np.matmul(temp, generating_reflections[k])
					temp = np.matmul(temp, generating_reflections[l])
					temp = np.matmul(temp, generating_reflections[m])
					temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue					
						temp = np.matmul(generating_reflections[i], generating_reflections[j])
						temp = np.matmul(temp, generating_reflections[k])
						temp = np.matmul(temp, generating_reflections[l])
						temp = np.matmul(temp, generating_reflections[m])
						temp = np.matmul(temp, generating_reflections[n])
						temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue			
							temp = np.matmul(generating_reflections[i], generating_reflections[j])
							temp = np.matmul(temp, generating_reflections[k])
							temp = np.matmul(temp, generating_reflections[l])
							temp = np.matmul(temp, generating_reflections[m])
							temp = np.matmul(temp, generating_reflections[n])
							temp = np.matmul(temp, generating_reflections[o])
							temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue	
							for p in range(3):
								if o == p:
									continue
								if ([m, n, o, p]==[0,2,0,2]) or ([m, n, o, p]==[2,0,2,0]):
									continue
								temp = np.matmul(generating_reflections[i], generating_reflections[j])
								temp = np.matmul(temp, generating_reflections[k])
								temp = np.matmul(temp, generating_reflections[l])
								temp = np.matmul(temp, generating_reflections[m])
								temp = np.matmul(temp, generating_reflections[n])
								temp = np.matmul(temp, generating_reflections[o])
								temp = np.matmul(temp, generating_reflections[p])
								temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue	
							for p in range(3):
								if o == p:
									continue
								if ([m, n, o, p]==[0,2,0,2]) or ([m, n, o, p]==[2,0,2,0]):
									continue
								for q in range(3):	
									if p == q:
										continue
									if ([n, o, p, q]==[0,2,0,2]) or ([n, o, p, q]==[2,0,2,0]):
										continue	
									temp = np.matmul(generating_reflections[i], generating_reflections[j])
									temp = np.matmul(temp, generating_reflections[k])
									temp = np.matmul(temp, generating_reflections[l])
									temp = np.matmul(temp, generating_reflections[m])
									temp = np.matmul(temp, generating_reflections[n])
									temp = np.matmul(temp, generating_reflections[o])
									temp = np.matmul(temp, generating_reflections[p])
									temp = np.matmul(temp, generating_reflections[q])
									temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue	
							for p in range(3):
								if o == p:
									continue
								if ([m, n, o, p]==[0,2,0,2]) or ([m, n, o, p]==[2,0,2,0]):
									continue
								for q in range(3):	
									if p == q:
										continue
									if ([n, o, p, q]==[0,2,0,2]) or ([n, o, p, q]==[2,0,2,0]):
										continue
									for r in range(3):	
										if q == r:
											continue
										if ([o, p, q, r]==[0,2,0,2]) or ([o, p, q, r]==[2,0,2,0]):
											continue								
										temp = np.matmul(generating_reflections[i], generating_reflections[j])
										temp = np.matmul(temp, generating_reflections[k])
										temp = np.matmul(temp, generating_reflections[l])
										temp = np.matmul(temp, generating_reflections[m])
										temp = np.matmul(temp, generating_reflections[n])
										temp = np.matmul(temp, generating_reflections[o])
										temp = np.matmul(temp, generating_reflections[p])
										temp = np.matmul(temp, generating_reflections[q])
										temp = np.matmul(temp, generating_reflections[r])
										temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue	
							for p in range(3):
								if o == p:
									continue
								if ([m, n, o, p]==[0,2,0,2]) or ([m, n, o, p]==[2,0,2,0]):
									continue
								for q in range(3):	
									if p == q:
										continue
									if ([n, o, p, q]==[0,2,0,2]) or ([n, o, p, q]==[2,0,2,0]):
										continue
									for r in range(3):	
										if q == r:
											continue
										if ([o, p, q, r]==[0,2,0,2]) or ([o, p, q, r]==[2,0,2,0]):
											continue
										for s in range(3):	
											if r == s:
												continue
											if ([p, q, r, s]==[0,2,0,2]) or ([p, q, r, s]==[2,0,2,0]):
												continue	
											temp = np.matmul(generating_reflections[i], generating_reflections[j])
											temp = np.matmul(temp, generating_reflections[k])
											temp = np.matmul(temp, generating_reflections[l])
											temp = np.matmul(temp, generating_reflections[m])
											temp = np.matmul(temp, generating_reflections[n])
											temp = np.matmul(temp, generating_reflections[o])
											temp = np.matmul(temp, generating_reflections[p])
											temp = np.matmul(temp, generating_reflections[q])
											temp = np.matmul(temp, generating_reflections[r])
											temp = np.matmul(temp, generating_reflections[s])
											temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue	
							for p in range(3):
								if o == p:
									continue
								if ([m, n, o, p]==[0,2,0,2]) or ([m, n, o, p]==[2,0,2,0]):
									continue
								for q in range(3):	
									if p == q:
										continue
									if ([n, o, p, q]==[0,2,0,2]) or ([n, o, p, q]==[2,0,2,0]):
										continue
									for r in range(3):	
										if q == r:
											continue
										if ([o, p, q, r]==[0,2,0,2]) or ([o, p, q, r]==[2,0,2,0]):
											continue
										for s in range(3):	
											if r == s:
												continue
											if ([p, q, r, s]==[0,2,0,2]) or ([p, q, r, s]==[2,0,2,0]):
												continue
											for t in range(3):
												if s == t:
													continue
												if ([q, r, s, t]==[0,2,0,2]) or ([q, r, s, t]==[2,0,2,0]):
													continue	
												temp = np.matmul(generating_reflections[i], generating_reflections[j])
												temp = np.matmul(temp, generating_reflections[k])
												temp = np.matmul(temp, generating_reflections[l])
												temp = np.matmul(temp, generating_reflections[m])
												temp = np.matmul(temp, generating_reflections[n])
												temp = np.matmul(temp, generating_reflections[o])
												temp = np.matmul(temp, generating_reflections[p])
												temp = np.matmul(temp, generating_reflections[q])
												temp = np.matmul(temp, generating_reflections[r])
												temp = np.matmul(temp, generating_reflections[s])
												temp = np.matmul(temp, generating_reflections[t])
												temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue	
							for p in range(3):
								if o == p:
									continue
								if ([m, n, o, p]==[0,2,0,2]) or ([m, n, o, p]==[2,0,2,0]):
									continue
								for q in range(3):	
									if p == q:
										continue
									if ([n, o, p, q]==[0,2,0,2]) or ([n, o, p, q]==[2,0,2,0]):
										continue
									for r in range(3):	
										if q == r:
											continue
										if ([o, p, q, r]==[0,2,0,2]) or ([o, p, q, r]==[2,0,2,0]):
											continue
										for s in range(3):	
											if r == s:
												continue
											if ([p, q, r, s]==[0,2,0,2]) or ([p, q, r, s]==[2,0,2,0]):
												continue
											for t in range(3):
												if s == t:
													continue
												if ([q, r, s, t]==[0,2,0,2]) or ([q, r, s, t]==[2,0,2,0]):
													continue
												for u in range(3):
													if t == u:
														continue
													if ([r, s, t, u]==[0,2,0,2]) or ([r, s, t, u]==[2,0,2,0]):
														continue	
													temp = np.matmul(generating_reflections[i], generating_reflections[j])
													temp = np.matmul(temp, generating_reflections[k])
													temp = np.matmul(temp, generating_reflections[l])
													temp = np.matmul(temp, generating_reflections[m])
													temp = np.matmul(temp, generating_reflections[n])
													temp = np.matmul(temp, generating_reflections[o])
													temp = np.matmul(temp, generating_reflections[p])
													temp = np.matmul(temp, generating_reflections[q])
													temp = np.matmul(temp, generating_reflections[r])
													temp = np.matmul(temp, generating_reflections[s])
													temp = np.matmul(temp, generating_reflections[t])
													temp = np.matmul(temp, generating_reflections[u])
													temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue	
							for p in range(3):
								if o == p:
									continue
								if ([m, n, o, p]==[0,2,0,2]) or ([m, n, o, p]==[2,0,2,0]):
									continue
								for q in range(3):	
									if p == q:
										continue
									if ([n, o, p, q]==[0,2,0,2]) or ([n, o, p, q]==[2,0,2,0]):
										continue
									for r in range(3):	
										if q == r:
											continue
										if ([o, p, q, r]==[0,2,0,2]) or ([o, p, q, r]==[2,0,2,0]):
											continue
										for s in range(3):	
											if r == s:
												continue
											if ([p, q, r, s]==[0,2,0,2]) or ([p, q, r, s]==[2,0,2,0]):
												continue
											for t in range(3):
												if s == t:
													continue
												if ([q, r, s, t]==[0,2,0,2]) or ([q, r, s, t]==[2,0,2,0]):
													continue
												for u in range(3):
													if t == u:
														continue
													if ([r, s, t, u]==[0,2,0,2]) or ([r, s, t, u]==[2,0,2,0]):
														continue
													for v in range(3):
														if u == v:
															continue
														if ([s, t, u, v]==[0,2,0,2]) or ([s, t, u, v]==[2,0,2,0]):
															continue
														temp = np.matmul(generating_reflections[i], generating_reflections[j])
														temp = np.matmul(temp, generating_reflections[k])
														temp = np.matmul(temp, generating_reflections[l])
														temp = np.matmul(temp, generating_reflections[m])
														temp = np.matmul(temp, generating_reflections[n])
														temp = np.matmul(temp, generating_reflections[o])
														temp = np.matmul(temp, generating_reflections[p])
														temp = np.matmul(temp, generating_reflections[q])
														temp = np.matmul(temp, generating_reflections[r])
														temp = np.matmul(temp, generating_reflections[s])
														temp = np.matmul(temp, generating_reflections[t])
														temp = np.matmul(temp, generating_reflections[u])
														temp = np.matmul(temp, generating_reflections[v])
														temp = np.round_(temp, decimals = precision)
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
		if i == j:
			continue
		for k in range(3):
			if j == k:
				continue
			for l in range(3):
				if k == l:
					continue
				if ([i, j, k, l]==[0,2,0,2]) or ([i, j, k, l]==[2,0,2,0]):
					continue
				for m in range(3):	
					if l == m:
						continue
					if ([j, k, l, m]==[0,2,0,2]) or ([j, k, l, m]==[2,0,2,0]):
						continue	
					for n in range(3):
						if m == n:
							continue
						if ([k, l, m, n]==[0,2,0,2]) or ([k, l, m, n]==[2,0,2,0]):
							continue				
						for o in range(3):
							if n == o:
								continue
							if ([l, m, n, o]==[0,2,0,2]) or ([l, m, n, o]==[2,0,2,0]):
								continue	
							for p in range(3):
								if o == p:
									continue
								if ([m, n, o, p]==[0,2,0,2]) or ([m, n, o, p]==[2,0,2,0]):
									continue
								for q in range(3):	
									if p == q:
										continue
									if ([n, o, p, q]==[0,2,0,2]) or ([n, o, p, q]==[2,0,2,0]):
										continue
									for r in range(3):	
										if q == r:
											continue
										if ([o, p, q, r]==[0,2,0,2]) or ([o, p, q, r]==[2,0,2,0]):
											continue
										for s in range(3):	
											if r == s:
												continue
											if ([p, q, r, s]==[0,2,0,2]) or ([p, q, r, s]==[2,0,2,0]):
												continue
											for t in range(3):
												if s == t:
													continue
												if ([q, r, s, t]==[0,2,0,2]) or ([q, r, s, t]==[2,0,2,0]):
													continue
												for u in range(3):
													if t == u:
														continue
													if ([r, s, t, u]==[0,2,0,2]) or ([r, s, t, u]==[2,0,2,0]):
														continue
													for v in range(3):
														if u == v:
															continue
														if ([s, t, u, v]==[0,2,0,2]) or ([s, t, u, v]==[2,0,2,0]):
															continue
														for w in range(3):
															if v == w:
																continue
															if ([t, u, v, w]==[0,2,0,2]) or ([t, u, v, w]==[2,0,2,0]):
																continue
														temp = np.matmul(generating_reflections[i], generating_reflections[j])
														temp = np.matmul(temp, generating_reflections[k])
														temp = np.matmul(temp, generating_reflections[l])
														temp = np.matmul(temp, generating_reflections[m])
														temp = np.matmul(temp, generating_reflections[n])
														temp = np.matmul(temp, generating_reflections[o])
														temp = np.matmul(temp, generating_reflections[p])
														temp = np.matmul(temp, generating_reflections[q])
														temp = np.matmul(temp, generating_reflections[r])
														temp = np.matmul(temp, generating_reflections[s])
														temp = np.matmul(temp, generating_reflections[t])
														temp = np.matmul(temp, generating_reflections[u])
														temp = np.matmul(temp, generating_reflections[v])
														temp = np.matmul(temp, generating_reflections[w])
														temp = np.round_(temp, decimals = precision)
														length = len(group_elements)
														for index in range(length):
															if np.array_equal(group_elements[index], temp):
																break
															if index == length - 1:
																group_elements.append(temp)
																print(temp)
print(len(group_elements))
