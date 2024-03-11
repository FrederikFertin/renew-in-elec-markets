import pandapower.networks as nw
import pandapower.plotting as plot
import numpy as np
import pandapower as pp
from natsort import natsorted
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import csv
from matplotlib.colors import Normalize
from matplotlib import rcParams
import datetime as dt

 

def createNetwork(mapping_gen, mapping_loads, mapping_wind):
    # create empty net
    net = pp.create_empty_network()

    #bus_map = pd.read_csv('Assignment 1/bus_map.csv', delimiter=';')
    bus_map = pd.read_csv('bus_map.csv', delimiter=';')
    
    #line_map = pd.read_csv('Assignment 1/lines.csv', delimiter=';')
    line_map = pd.read_csv('lines.csv', delimiter=';')
    
    # Create buses
    for i in range(len(bus_map)):
        pp.create_bus(net, vn_kv=0.4, index = i,
                      geodata=(bus_map['x-coord'][i], -bus_map['y-coord'][i]),
                      name=bus_map['Bus'][i])
        
        for j in range(len(mapping_gen[bus_map['Bus'][i]])):
            pp.create_gen(net, bus=i, p_mw=100)
        for j in range(len(mapping_loads[bus_map['Bus'][i]])):
            pp.create_load(net, bus=i, p_mw=100)
        for j in range(len(mapping_wind[bus_map['Bus'][i]])):
            pp.create_ext_grid(net, bus=i, p_mw=100, vm_pu=1.05)
        

    # Create lines
    for i in range(len(line_map)):
        pp.create_line_from_parameters(net,
                from_bus=    int(line_map['FromBus'][i][1:])-1,
                to_bus=     int(line_map['ToBus'][i][1:])-1,
                length_km=2,
                name='L'+str(i+1),
                r_ohm_per_km=0.2,
                x_ohm_per_km=0.07,
                c_nf_per_km=0.3,
                max_i_ka=100)
    
    return net


def drawNormal(net):
    
    bus_map = pd.read_csv('bus_map.csv', delimiter=';')
    
    line_map = pd.read_csv('lines.csv', delimiter=';')
        
    size = 5
    
    d_c = plot.create_load_collection(net, loads=net.load.index, size=size)
    gen_c = plot.create_gen_collection(net, gens=net.gen.index, size=size, orientation=0)
    wind_c = plot.create_ext_grid_collection(net, ext_grids=net.ext_grid.index, size=size, orientation=3.14/2)
    
    bc = plot.create_bus_collection(net, buses=net.bus.index, size=size, 
                                    zorder=10, color='blue')
    
    lc = plot.create_line_collection(net, lines=net.line.index, zorder=1, use_bus_geodata=True, color='grey')
    
    plot.draw_collections([lc, d_c, gen_c, wind_c, bc])
    plt.title("Network", fontsize=30)
    plt.legend()
    plt.show()
    
def drawSingleStep(net, p_G, p_W, LMP):
    
    size = 5
    
    cmap = plt.get_cmap('rainbow')
    norm_bus = Normalize(0,25)
    
    d_c = plot.create_load_collection(net, loads=net.load.index, size=size) 
    gen_c = plot.create_gen_collection(net, gens=net.gen.index, size=size, orientation=0)
    wind_c = plot.create_ext_grid_collection(net, ext_grids=net.ext_grid.index, size=size, orientation=3.14/2)
    
    lc = plot.create_line_collection(net, lines=net.line.index, zorder=1,\
                        use_bus_geodata=True,color='grey')
    
    bc = plot.create_bus_collection(net, buses=net.bus.index, size=size, 
            zorder=1, z=list(LMP.values()), cmap=cmap, norm=norm_bus, cbar_title="Node LMP [DKK/MWh]")# ,use_bus_geodata=True)
    
    plot.draw_collections([d_c, gen_c, wind_c, lc, bc])
    plt.title('Single time-step DC OPF', fontsize=20)
    plt.show()


def drawLMP(net, lambda_):
    
    size = 5
    
    d_c = plot.create_load_collection(net, loads=net.load.index, size=size)
    gen_c = plot.create_gen_collection(net, gens=net.gen.index, size=size, orientation=0)
    wind_c = plot.create_ext_grid_collection(net, ext_grids=net.ext_grid.index, size=size, orientation=3.14/2)
    
    lc = plot.create_line_collection(net, lines=net.line.index, zorder=1,\
                        use_bus_geodata=True,color='grey')
        
    for t, lambdas_t in lambda_.items():
        
        lmp_t = list(lambdas_t.values())
        
        cmap = plt.get_cmap('rainbow')
        norm = Normalize(4,14)
        bc = plot.create_bus_collection(net, buses=net.bus.index, size=size, 
                zorder=1, z=lmp_t, cmap=cmap, norm=norm, cbar_title="Node LMP [DKK]")# ,use_bus_geodata=True)
        
        plot.draw_collections([d_c, gen_c, wind_c, lc, bc])
        plt.title('Network LMPs ' + str(t), fontsize=20)
        plt.show()





























# %%
