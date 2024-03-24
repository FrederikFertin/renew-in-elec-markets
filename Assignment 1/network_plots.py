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
    """ Initializes a pandapower network based on the given mappings of generators,
        loads and wind turbines to buses."""

    # create empty net
    net = pp.create_empty_network()
    cwd = os.path.dirname(__file__)
    bus_map = pd.read_csv(cwd + '/input_data/bus_map.csv', delimiter=';')
    
    line_map = pd.read_csv(cwd + '/input_data/lines.csv', delimiter=';')
    
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
            pp.create_sgen(net, bus=i, p_mw=100)#, vm_pu=1.05)
        

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
    """ Draws the network with buses, lines, generators, loads and wind turbines. """
    
    size = 5
    
    d_c = plot.create_load_collection(net, loads=net.load.index, size=size)
    gen_c = plot.create_gen_collection(net, gens=net.gen.index, size=size, orientation=0)
    wind_c = plot.create_sgen_collection(net, sgens=net.sgen.index, size=size, orientation=3.14/2)
    
    bc = plot.create_bus_collection(net, buses=net.bus.index, size=size, 
                                    zorder=10, color='blue')
    
    lc = plot.create_line_collection(net, lines=net.line.index, zorder=1, use_bus_geodata=True, color='grey')
    
    plot.draw_collections([lc, d_c, gen_c, wind_c, bc])
    plt.title("Network", fontsize=30)
    plt.legend()
    plt.show()


def drawLMP(net, lambda_, loading: dict | None = None):
    """ Draws the network with the LMPs of each bus as a color gradient.
        Also highlights congested lines if loading is provided. """

    size = 5
    draw_collection = []
    draw_collection.append(plot.create_load_collection(net, loads=net.load.index, size=size))
    draw_collection.append(plot.create_gen_collection(net, gens=net.gen.index, size=size, orientation=0))
    draw_collection.append(plot.create_sgen_collection(net, sgens=net.sgen.index, size=size, orientation=3.14/2))
    draw_collection.append(plot.create_line_collection(net, lines=net.line.index, zorder=1, use_bus_geodata=True, color='grey'))
    
    buses = net.bus.index.tolist() # list of all bus indices
    coords = zip(net.bus_geodata.x.loc[buses].values - 25, net.bus_geodata.y.loc[buses].values) # tuples of all bus coords
    bic = plot.create_annotation_collection(size=10, texts=net.bus.name, coords = coords, zorder=3, color='grey')
    draw_collection.append(bic)

    cwd = os.path.dirname(__file__)
    line_map = pd.read_csv(cwd + '/lines.csv', delimiter=';')

    for t, lambdas_t in lambda_.items():
        draw_collection_t = draw_collection.copy()
        lmp_t = list(lambdas_t.values())
        
        cmap = plt.get_cmap('rainbow')
        norm = Normalize(0,20)
        bc = plot.create_bus_collection(net, buses=net.bus.index, size=size, 
                zorder=1, z=lmp_t, cmap=cmap, norm=norm, cbar_title="Node LMP [$/MWh]")# ,use_bus_geodata=True)
        draw_collection_t.append(bc)

        if loading is not None:
            for n, connections in loading[t].items():
                for m, dual  in connections.items():
                    if dual != 0:
                        row1 = line_map.loc[(line_map['FromBus'] == n).values * (line_map['ToBus'] == m).values]
                        row2 = line_map.loc[(line_map['FromBus'] == m).values * (line_map['ToBus'] == n).values]
                        if not row1.empty:
                            line_ix = row1.index[0]
                        elif not row2.empty:
                            line_ix = row2.index[0]
                        
                        lc = plot.create_line_collection(net, lines=[line_ix], zorder=2, use_bus_geodata=True, color='red')
                        draw_collection_t.append(lc)

        plot.draw_collections(draw_collection_t)
        plt.title('Network LMPs ' + str(t), fontsize=20)
        plt.show()

def drawTheta(net, theta_, loading: dict | None = None):
    """ Draws the network with the voltage angles of each bus as a color gradient. 
        Also highlights congested lines if loading is provided. """
    
    size = 5
    draw_collection = []
    draw_collection.append(plot.create_load_collection(net, loads=net.load.index, size=size))
    draw_collection.append(plot.create_gen_collection(net, gens=net.gen.index, size=size, orientation=0))
    draw_collection.append(plot.create_sgen_collection(net, sgens=net.sgen.index, size=size, orientation=3.14/2))
    draw_collection.append(plot.create_line_collection(net, lines=net.line.index, zorder=1, use_bus_geodata=True, color='grey'))
    

    buses = net.bus.index.tolist() # list of all bus indices
    coords = zip(net.bus_geodata.x.loc[buses].values - 25, net.bus_geodata.y.loc[buses].values) # tuples of all bus coords
    bic = plot.create_annotation_collection(size=10, texts=net.bus.name, coords = coords, zorder=3, color='grey')
    draw_collection.append(bic)

    cwd = os.path.dirname(__file__)
    line_map = pd.read_csv(cwd + '/lines.csv', delimiter=';')
        
    for t, thetas_t in theta_.items():
        draw_collection_t = draw_collection.copy()
        u_t = list(thetas_t.values())
        
        cmap = plt.get_cmap('rainbow')
        norm = Normalize(-0.5, 2.5)
        draw_collection_t.append(plot.create_bus_collection(net, buses=net.bus.index, size=size, 
                zorder=1, z=u_t, cmap=cmap, norm=norm, cbar_title="Node Voltage Angle [Rad]"))# ,use_bus_geodata=True)

        if loading is not None:
            for n, connections in loading[t].items():
                for m, dual  in connections.items():
                    if dual != 0:
                        row1 = line_map.loc[(line_map['FromBus'] == n).values * (line_map['ToBus'] == m).values]
                        row2 = line_map.loc[(line_map['FromBus'] == m).values * (line_map['ToBus'] == n).values]
                        if not row1.empty:
                            line_ix = row1.index[0]
                        elif not row2.empty:
                            line_ix = row2.index[0]
                        
                        lc = plot.create_line_collection(net, lines=[line_ix], zorder=2, use_bus_geodata=True, color='red')
                        draw_collection_t.append(lc)


        plot.draw_collections(draw_collection_t)
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
