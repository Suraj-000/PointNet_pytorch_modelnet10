import trimesh
import os 
from path import Path

path=Path("../ModelNet10")
folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)}
print(classes)

tmesh = trimesh.load("../ModelNet10/sofa/train/sofa_0002.off")
scene = trimesh.scene.Scene()
scene.add_geometry(tmesh)
# scene.show()


tpcd=trimesh.load("../ModelNet10/sofa/train/sofa_0002.off")
sampled_pc = trimesh.sample.sample_surface(tpcd,10000)
pc = sampled_pc[0]
p= trimesh.points.PointCloud(pc)
scene = trimesh.scene.Scene()
scene.add_geometry(p)
scene.show()

