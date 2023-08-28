import trimesh
import numpy as np
import os 
import random
import torch
import math

class SamplePoints:
    def randomSample(self):
        indexes = random.sample(range(self.points.shape[0]), self.npoint)
        return self.points[indexes]

    def farthestPointSample(self):
        N, D = self.points.shape
        xyz = self.points[:,:3]
        centroids = np.zeros((self.npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(self.npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        points = self.points[centroids.astype(np.int32)]
        return points

    def normalize(self,spc):
        npc = spc - np.mean(spc, axis=0)
        npc /= np.max(np.linalg.norm(npc, axis=1))
        return npc

    def totensor(self,p):
        return torch.from_numpy(p)

    def RandomRotation(self,pc):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        rot_pointcloud = rot_matrix.dot(pc.T).T
        return  rot_pointcloud

    def RandomNoise(self,pc):
        noise = np.random.normal(0, 0.02, (pc.shape))
        npc = pc + noise
        return  npc

    def __init__(self,points,npoints=1024):
        self.points=points
        self.npoint=npoints