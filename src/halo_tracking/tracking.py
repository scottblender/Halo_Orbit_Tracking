"""Relative optical angles."""
import numpy as np

def relative_angles(relative_position,relative_velocity):
    """Return alpha/delta in rad and their rates in rad/s."""
    x,y,z=relative_position; vx,vy,vz=relative_velocity
    horizontal=np.hypot(x,y); squared_range=relative_position@relative_position
    if horizontal<1e-10 or squared_range<1e-20:
        raise ValueError('Observer-target angular coordinates are singular')
    return np.array([np.arctan2(y,x),np.arctan2(z,horizontal),
                     (x*vy-y*vx)/horizontal**2,
                     (horizontal*vz-z*(x*vx+y*vy)/horizontal)/squared_range])

