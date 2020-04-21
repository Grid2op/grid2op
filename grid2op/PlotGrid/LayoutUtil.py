import networkx as nx
import numpy as np
import copy

def layout_obs_sub_only(obs, scale=1000.0):
    n_sub = obs.n_sub
    n_line = obs.n_line
    or_sub = obs.line_or_to_subid
    ex_sub = obs.line_ex_to_subid
    
    # Create a graph of substations vertices
    G = nx.Graph()
    
    # Set lines edges
    for line_idx in range(n_line):
        lor_sub = or_sub[line_idx]
        lex_sub = ex_sub[line_idx]
        
        # Compute edge vertices indices for current graph
        left_v = lor_sub
        right_v = lex_sub
        
        # Register edge in graph
        G.add_edge(left_v, right_v)
        
    # Convert our layout to nx format
    initial_layout = {}
    for sub_idx, sub_name in enumerate(obs.name_sub):
        initial_layout[sub_idx] = obs.grid_layout[sub_name]
        
    # Use kamada_kawai algorithm
    kkl = nx.kamada_kawai_layout(G, scale=scale)
    # Convert back to our layout format
    improved_layout = {}
    for sub_idx, v in kkl.items():
        sub_key = obs.name_sub[sub_idx]
        vx = int(np.round(v[0]))
        vy = int(np.round(v[1]))
        improved_layout[sub_key] = [vx, vy]

    return improved_layout

def layout_obs_sub_load_and_gen(obs, scale=1000.0, use_initial=False):
    # Create a graph of substations vertices
    G = nx.Graph()

    sub_w = 0 if use_initial else 100
    load_w = 20 if use_initial else 15
    gen_w = 20 if use_initial else 25

    # Set lines edges
    for line_idx in range(obs.n_line):
        lor_sub = obs.line_or_to_subid[line_idx]
        lex_sub = obs.line_ex_to_subid[line_idx]
        
        # Compute edge vertices indices for current graph
        left_v = lor_sub
        right_v = lex_sub
        
        # Register edge in graph
        G.add_edge(left_v, right_v, weight=sub_w)

    # Set edges for loads
    load_offset = obs.n_sub
    for load_idx in range(obs.n_load):
        load_sub = obs.load_to_subid[load_idx]

        left_v = load_sub
        right_v = load_offset + load_idx
        # Register edge
        G.add_edge(left_v, right_v, weight=load_w)

    # Set edges for gens
    gen_offset = obs.n_sub + obs.n_load
    for load_idx in range(obs.n_gen):
        gen_sub = obs.gen_to_subid[load_idx]

        left_v = gen_sub
        right_v = gen_offset + load_idx
        # Register edge
        G.add_edge(left_v, right_v, weight=gen_w)

    # Convert our layout to nx format
    layout_keys = list(obs.grid_layout.keys())
    if use_initial:
        initial_layout = {}
        for sub_idx, sub_name in enumerate(layout_keys):
            sub_pos = copy.deepcopy(obs.grid_layout[sub_name])
            #sub_pos[0] *= scale
            #sub_pos[1] *= scale
            initial_layout[sub_idx] = sub_pos
        for load_idx, load_subid in enumerate(obs.load_to_subid):
            sub_name = layout_keys[load_subid]
            load_pos = list(copy.deepcopy(obs.grid_layout[sub_name]))
            load_pos[0] -= (load_idx + 1)
            load_pos[1] += (load_idx + 1)
            initial_layout[load_offset + load_idx] = load_pos
        for gen_idx, gen_subid in enumerate(obs.gen_to_subid):
            sub_name = layout_keys[gen_subid]
            gen_pos = list(copy.deepcopy(obs.grid_layout[sub_name]))
            gen_pos[0] += (gen_idx + 1)
            gen_pos[1] -= (gen_idx + 1)
            initial_layout[gen_offset + gen_idx] = gen_pos        
    else:
        initial_layout = None

    if use_initial:
        fix = list(range(obs.n_sub))
        seed = np.random.RandomState(0)
        # Use Fruchterman-Reingold algorithm
        kkl = nx.spring_layout(G, scale=scale, fixed=fix, pos=initial_layout, seed=seed, iterations=500)
    else:
        # Use kamada_kawai algorithm
        kkl = nx.kamada_kawai_layout(G, scale=scale)

    # Convert back to our layout format
    improved_layout = {}
    for sub_idx, sub_name in enumerate(layout_keys):
        key = sub_name
        v = kkl[sub_idx]
        vx = np.round(v[0])
        vy = np.round(v[1])
        improved_layout[key] = [vx, vy]    
    for load_idx, load_subid in enumerate(obs.load_to_subid):
        key = obs.name_load[load_idx]
        v = kkl[load_offset + load_idx]
        vx = np.round(v[0])
        vy = np.round(v[1])
        improved_layout[key] = [vx, vy]
    for gen_idx, gen_subid in enumerate(obs.gen_to_subid):
        key = obs.name_gen[gen_idx]
        v = kkl[gen_offset + gen_idx]
        vx = np.round(v[0])
        vy = np.round(v[1])
        improved_layout[key] = [vx, vy]

    return improved_layout
