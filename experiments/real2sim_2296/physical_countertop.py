"""Continuous beveled countertop with a through-aperture and triangle collision."""
import numpy as np

def outline(bounds,radius):
    x0,y0,x1,y1=map(float,bounds);r=float(radius)
    if not 0<2*r<min(x1-x0,y1-y0):raise ValueError('Invalid corner radius')
    return np.array([[x0+r,y0],[x1-r,y0],[x1,y0+r],[x1,y1-r],
                     [x1-r,y1],[x0+r,y1],[x0,y1-r],[x0,y0+r]])

def slab(width,depth,height,cutout,thickness=.032,bevel=.003):
    """Return one outward-wound static mesh; dimensions are nominal meters.

    top_right is retained as the existing countertop support-test prim path.
    It now names the whole continuous slab instead of only its right section.
    """
    width,depth,height,thickness,bevel=map(float,[width,depth,height,thickness,bevel])
    cx,cy,cw,cd=map(float,cutout)
    if not np.isfinite([width,depth,height,thickness,bevel,cx,cy,cw,cd]).all():
        raise ValueError('All slab dimensions must be finite')
    outer=np.array([-width/2,0.,width/2,depth]);inner=np.array([cx-cw/2,cy-cd/2,cx+cw/2,cy+cd/2])
    if not (outer[0]+2*bevel<inner[0]<inner[2]<outer[2]-2*bevel and
            outer[1]+2*bevel<inner[1]<inner[3]<outer[3]-2*bevel and 2*bevel<thickness):
        raise ValueError('Sink opening must lie inside the slab')
    corner=2*bevel;vertices=[];outer_rings=[];inner_rings=[]
    zs=[height-thickness,height-thickness+bevel,height-bevel,height]
    for side,bounds,rings in [('outer',outer,outer_rings),('inner',inner,inner_rings)]:
        for level,z in enumerate(zs):
            edge=bevel if level in (0,3) else 0.
            inset=edge if side=='outer' else -edge
            rect=bounds+np.array([inset,inset,-inset,-inset])
            points=outline(rect,corner-inset)
            ids=list(range(len(vertices),len(vertices)+len(points)));rings.append(ids)
            vertices.extend(np.c_[points,np.full(len(points),z)].tolist())
    faces=[]
    def quad(q,reverse=False):
        if reverse:q=q[::-1]
        faces.extend([[q[0],q[1],q[2]],[q[0],q[2],q[3]]])
    for rings,reverse in [(outer_rings,False),(inner_rings,True)]:
        for level in range(3):
            for k in range(8):
                n=(k+1)%8
                quad([rings[level][k],rings[level][n],rings[level+1][n],rings[level+1][k]],reverse)
    for level,reverse in [(3,False),(0,True)]:
        for k in range(8):
            n=(k+1)%8
            quad([outer_rings[level][k],outer_rings[level][n],inner_rings[level][n],inner_rings[level][k]],reverse)
    v=np.asarray(vertices);f=np.asarray(faces)
    # One continuous texture domain removes patch-edge exposure discontinuities.
    uv=np.c_[(v[:,0]+width/2)/width,v[:,1]/depth][f]
    return dict(name='top_right',vertices=vertices,faces=faces,uv=uv.tolist(),
                material='white',body='base',collision=True,collision_approximation='none')
