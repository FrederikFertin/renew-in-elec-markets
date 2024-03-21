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
import os

 

def createNetwork(mapping_gen, mapping_loads, mapping_wind):
    # create empty net
    net = pp.create_empty_network()
    cwd = os.getcwd()
    bus_map = pd.read_csv(cwd + '/bus_map.csv', delimiter=';')
    #bus_map = pd.read_csv('bus_map.csv', delimiter=';')
    
    line_map = pd.read_csv(cwd + '/lines.csv', delimiter=';')
    #line_map = pd.read_csv('lines.csv', delimiter=';')
    
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
        norm = Normalize(0,20)
        bc = plot.create_bus_collection(net, buses=net.bus.index, size=size, 
                zorder=1, z=lmp_t, cmap=cmap, norm=norm, cbar_title="Node LMP [$/MWh]")# ,use_bus_geodata=True)
        
        plot.draw_collections([d_c, gen_c, wind_c, lc, bc])
        plt.title('Network LMPs ' + str(t), fontsize=20)
        plt.show()

def drawTheta(net, theta_):
    
    size = 5
    
    d_c = plot.create_load_collection(net, loads=net.load.index, size=size)
    gen_c = plot.create_gen_collection(net, gens=net.gen.index, size=size, orientation=0)
    wind_c = plot.create_ext_grid_collection(net, ext_grids=net.ext_grid.index, size=size, orientation=3.14/2)
    
    lc = plot.create_line_collection(net, lines=net.line.index, zorder=1,\
                        use_bus_geodata=True,color='grey')
        
    for t, thetas_t in theta_.items():
        
        u_t = list(thetas_t.values())
        
        cmap = plt.get_cmap('rainbow')
        norm = Normalize(-2, 4)
        bc = plot.create_bus_collection(net, buses=net.bus.index, size=size, 
                zorder=1, z=u_t, cmap=cmap, norm=norm, cbar_title="Node Voltage Angle [Rad]")# ,use_bus_geodata=True)
        
        plot.draw_collections([d_c, gen_c, wind_c, lc, bc])
        plt.title('Network Voltage Angles ' + str(t), fontsize=20)
        plt.show()

def plot_SD_curve(ec, T):
    """
    Constructs the supply and demand curves of a given hour based on the run economic dispatch.
    """

    sort_offers = [('G100', 100, sum(ec.P_G_max.values()))]
    for g, offer_volume in ec.P_G_max.items():
        for ix, gen_data in enumerate(sort_offers):
            if gen_data[1] > ec.C_G_offer[g]:
                sort_offers.insert(ix, (g, ec.C_G_offer[g], offer_volume))
                break
    
    sort_offers = sort_offers[0:len(sort_offers)-1]
    plt.plot([0,sum(ec.P_W[T].values())], [0, 0], linewidth=1, color='blue', label='Supply Curve')
    point = np.array([sum(ec.P_W[T].values()), 0])

    for i in sort_offers:
        up_point = np.array([point[0], i[1]])
        right_point = np.array([point[0] + i[2], i[1]])
        plt.plot([point[0], up_point[0]], [point[1], up_point[1]], linewidth=1, color='blue')
        plt.plot([up_point[0], right_point[0]], [up_point[1], right_point[1]], linewidth=1, color='blue')
        point = right_point.copy()
    
    sort_bids = [('D100', 0, sum(ec.P_D[T].values()))]
    for d, bid_volume in ec.P_D[T].items():
        for ix, demand_data in enumerate(sort_bids):
            if demand_data[1] < ec.U_D[T][d]:
                sort_bids.insert(ix, (d, ec.U_D[T][d], bid_volume))
                break
    
    sort_bids = sort_bids[0:len(sort_bids)-1]
    
    plt.plot([0,sort_bids[0][2]], [sort_bids[0][1], sort_bids[0][1]], linewidth=1, color='orange', label='Demand Curve')
    point = np.array([sort_bids[0][2], sort_bids[0][1]])

    for i in sort_bids:
        down_point = np.array([point[0], i[1]])
        right_point = np.array([point[0] + i[2], i[1]])
        plt.plot([point[0], down_point[0]], [point[1], down_point[1]], linewidth=1, color='orange')
        plt.plot([down_point[0], right_point[0]], [down_point[1], right_point[1]], linewidth=1, color='orange')
        point = right_point.copy()
    
    plt.plot([point[0], point[0]], [point[1], 0], linewidth=1, color='orange')
    plt.title("Supply and Demand from 07:00 to 08:00")
    plt.xlabel("Quantity [MWh]")
    plt.ylabel("Price [$/MWh]")
    plt.axhline(ec.data.lambda_[T], color = 'black', linewidth=0.5, linestyle='--', label='Electricity Price')
    plt.legend()
    plt.show()



























# %%
