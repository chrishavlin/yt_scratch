# test case for adjusting plot_lines 
# python adjustable_plotlines.py 

# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pyglet
from pyglet import gl

import imgui
import numpy as np 
from imgui.integrations.pyglet import create_renderer

class TFchannel(object):
    def __init__(self,x_c=100):
        self.R_x = np.linspace(0,255,256) 
        self.R_y = self.gauss(mag = 0.)
        self.slider_val=10.
        self.slider_sensitivity=1.
        self.Data=np.random.normal(loc=10,scale=3,size=1000)
        self.HistData,self.HistBins=np.histogram(self.Data,bins=100)
        
    def gauss(self,x_c=100.,wid=10.,mag=1.):        
        return mag*np.exp( -((self.R_x - x_c)/ (wid))**2)
        
def main(TFch):

    window = pyglet.window.Window(width=1280, height=720, resizable=True)
    gl.glClearColor(1, 1, 1, 1)
    imgui.create_context()
    impl = create_renderer(window)
    xgraphsize=700
    ygraphsize=200 
    
    def update(dt):
        imgui.new_frame()        
        
        imgui.begin("line", True)
        mi = imgui.get_item_rect_min() # top left 
        ma = imgui.get_item_rect_max() # bottom right
        
        _, TFch.slider_val = imgui.slider_float('TF source width', TFch.slider_val, 0.5, 60.0, '%.2f', 1.0)        
        _, TFch.slider_sensitivity = imgui.slider_float('TF source sensitivity', TFch.slider_sensitivity, 0.0, 10.0, '%.2f', 1.0)        
        
        imgui.plot_histogram("N obs.",TFch.HistData.astype("f4"),graph_size=(xgraphsize,50))
        
        
        imgui.plot_lines("RTF",TFch.R_y.astype("f4") ,scale_min=0,scale_max=1.,graph_size=(xgraphsize, ygraphsize),)
        # render_gui()
        
        if imgui.is_item_hovered():
            (x,y)=imgui.get_mouse_pos() 
            # find center of gaussian source 
            x_c = (x - mi.x) / xgraphsize * TFch.R_x.max()
            # print(['imgui',x,y,x_c])
            if imgui.is_mouse_dragging(2):
                dx, dy = impl.io.mouse_delta
                if dy != 0:
                    # add a source term with magnitude scaled to dy 
                    # print(['imp',dx,dy,x_c])
                    TFch.R_y = TFch.R_y + TFch.gauss(x_c=x_c,wid=TFch.slider_val,mag=-TFch.slider_sensitivity*dy/ygraphsize)
                    TFch.R_y[TFch.R_y>1.]=1.
                    TFch.R_y[TFch.R_y<0.]=0
                            
        imgui.end()        

    @window.event
    def on_draw():
        update(1/60.0)
        window.clear()
        imgui.render()
        impl.render(imgui.get_draw_data())
        
    pyglet.app.run()
    impl.shutdown()

if __name__ == "__main__":
    TFobj=TFchannel()
    main(TFobj)
