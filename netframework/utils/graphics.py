import numpy as np
from visdom import Visdom
from skimage.transform import rescale

class VisdomSurface(object):
    """Plots to Visdom"""
    
    def __init__(self, vis, env_name='main'):
        self.viz = vis
        self.env = env_name
        self.surface = {}
    
    def show(self, var_name, split_name, M):
        if var_name not in self.surface:
            self.surface[var_name] = self.viz.surf(X=np.array([M]),
                env=self.env, 
                opts=dict(
                legend=[split_name],
                title=var_name
            ))
        else:
            self.viz.surf(X=np.array([M]), 
                env=self.env, 
                win=self.surface[var_name]
                )
                
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    
    def __init__(self, vis, env_name='main'):
        self.viz = vis
        self.env = env_name
        self.plots = {}
    
    def show(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), 
                env=self.env, 
                opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epoch',
                ylabel=var_name,
                ytype='log'
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), 
                env=self.env, 
                win=self.plots[var_name], 
                name=split_name,
                update='append'
                )



class VisdomScatter(object):
    """Scatter to Visdom"""
    
    def __init__(self, vis, env_name='main'):
        self.viz = vis
        self.env = env_name
        self.scatters = {}

    def show(self, X, Y, title, legend, markersize=10):        
        if title not in self.scatters: 
            self.scatters[title] = self.viz.scatter( X=X, Y=Y,
                    env=self.env,
                    opts=dict(
                    legend=legend,
                    markersize=markersize,
                    title=title
                    )
                )
        else:
            self.viz.scatter( X=X, Y=Y,
                    env=self.env,
                    win=self.scatters[title],
                    opts=dict(
                    legend=legend,
                    markersize=markersize,
                    title=title
                    )
                )


class HeatMapVisdom(object):
    """Heat Map to Visdom"""
    
    def __init__(self, vis, env_name='main'):
        self.vis = vis
        self.env = env_name
        self.hmaps = {}
    
    def show(self, title, image, colormap='Viridis',scale=1):
        if scale!=1:
            image=rescale(image,scale,preserve_range=True)
        
        image=np.flipud(image)

        if title not in self.hmaps:
            self.hmaps[title] = self.vis.heatmap(
                image, 
                env=self.env, 
                opts=dict(title=title,colormap=colormap
            ))
        else:
            self.vis.heatmap(
                image,
                env=self.env, 
                win=self.hmaps[title], 
                opts=dict(title=title,colormap=colormap
            ))

class TextVisdom(object):
    """Heat Map to Visdom"""
    
    def __init__(self, vis, env_name='main'):
        self.vis = vis
        self.env = env_name
        self.vtext = {}
    
    def show(self, title, customtext):
        if title not in self.vtext:
            self.vtext[title] = self.vis.text(
                text=customtext, 
                env=self.env, 
                opts=dict(title=title
            ))
        else:
            self.vis.text(
                text=customtext,
                env=self.env, 
                win=self.vtext[title], 
                opts=dict(title=title
            ))
            
class ImageVisdom(object):
    """Images to Visdom"""
    
    def __init__(self, vis, env_name='main'):
        self.vis = vis
        self.env = env_name
        self.images = {}
    
    def show(self, title, image):
        if title not in self.images:
            self.images[title] = self.vis.image(
                image, 
                env=self.env, 
                opts=dict(title=title
            ))
            
        else:
            self.vis.image(
                image,
                env=self.env, 
                win=self.images[title], 
                opts=dict(title=title
            ))
