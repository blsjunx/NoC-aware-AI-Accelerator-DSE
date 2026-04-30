#!/usr/bin/env python3
"""
Lightweight DSE loop (binary-driven):
- Copies required inputs from manual workspace into workspaces/dse/<workspace>_<timestamp>/
- Modifies the copied stage_4_simulation/config.json each iteration
- Runs ./build/bin/tlm_sim directly (TLM mode only) for selected layer groups
- Logs every cur_config config, stdout/stderr, and aggregated results per iteration
"""

import argparse
import copy
import logging
import os
import shutil
import subprocess
import sys
import random
import numpy as np
import pandas as pd
import orjson
import re

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable


# Ensure project root is on sys.path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nettlm_dse.common.utils.utils import ConfigManager
from nettlm_dse.infrastructure.workspace import WorkspaceManager
from nettlm_dse.core.evaluation.tlm_simulator import _find_systemc_lib, SIMULATOR_PATH

VIS_TEMPLATE_PATH = Path("workspaces/manual/TP_AS25/stage_4_simulation/vis_report_config.json")

def config_selector(
    config: Dict[str, Any],
    iteration: int,
    allowed_groups: Optional[List[str]] = None,    
    last_iter_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    partitions = config.get("partitions")
    if not isinstance(partitions, dict):
        return config
    # Load last iteration logs
    
    if last_iter_dir.exists() & (iteration != 1):
        log_dir_path = last_iter_dir / "logs/layer_group_A"
        sim_result_path = last_iter_dir / "simulation_result.json"
        log_alloc_path = log_dir_path / "allocation.csv"
        log_exec_time_path = log_dir_path / "execution_time.csv"
        log_flowID_path = log_dir_path / "flowID.csv"
        log_link_load_path = log_dir_path / "link_load.csv"
        log_routers_path = list(log_dir_path.glob("router_*.csv"))
        log_router_path = [f for f in log_routers_path if re.match(r'^router_\d+\.csv$', f.name)]
        log_router_buf_path = [f for f in log_routers_path if re.match(r'^router_\d+_buf\.csv$', f.name)]
        log_d2d_buf_path = list(log_dir_path.glob("d2d_*_*_buf.csv"))
        
        with open(sim_result_path, "rb") as f:
            sim_result = orjson.loads(f.read())
        throughput = sim_result['data']['summary']['total_throughput_ops']
        power_effs = sim_result['data']['summary']['power_efficiency_ops_per_w']
        alloc = pd.read_csv(log_alloc_path)
        exec_time = pd.read_csv(log_exec_time_path)
        flowID = pd.read_csv(log_flowID_path)
        link_load = pd.read_csv(log_link_load_path)
        routers = [pd.read_csv(file, skipinitialspace=True) for file in log_router_path]
        router_bufs = [pd.read_csv(file, dtype=str, skipinitialspace=True) for file in log_router_buf_path]
        d2d_bufs = [pd.read_csv(file, dtype=str, skipinitialspace=True) for file in log_d2d_buf_path]
    
    """ Edit code below """
    # -------------------------------------------------------------------------
    # [Final Hybrid DSE] 
    # Features:
    #   1. Analysis: Bidirectional Dependency (Parent & Child Gravity).
    #   2. Allocation: Adaptive Limit (Total/3), Dynamic Boosting, Universal Quantization.
    #   3. Placement: 
    #      - Split Isolation (Top/Bottom edges).
    #      - Heavy Node: Column-Major Fill (Left->Right).
    #      - Light Node: Vertical-Locked Random Gravity.
    #   4. Tensor: Full DRAM usage with Factorization.
    # -------------------------------------------------------------------------

    import json
    import math

    # --- [0] Helper Functions ---
    def get_coords(rid, w): return (rid % w, rid // w)
    def get_rid(x, y, w): return y * w + x

    def dist_manhattan(rid1, rid2, w):
        x1, y1 = get_coords(rid1, w)
        x2, y2 = get_coords(rid2, w)
        return abs(x1 - x2) + abs(y1 - y2)

    def dist_coords(c1, x2, y2, w):
        x1, y1 = get_coords(c1, w)
        return abs(x1 - x2) + abs(y1 - y2)

    def get_quadrant_pools(v_ids, gx, gy):
        mx = gx // 2
        my = gy // 2
        pools = [[], [], [], []] # TL, TR, BL, BR
        for rid in v_ids:
            x, y = get_coords(rid, gx)
            if x < mx and y < my:     q = 0
            elif x >= mx and y < my:  q = 1
            elif x < mx and y >= my:  q = 2
            else:                     q = 3
            pools[q].append(rid)
        return pools

    def get_symmetric_cores(seed_rid, gx, gy, valid_cores_set):
        sx, sy = get_coords(seed_rid, gx)
        mx, my = gx // 2, gy // 2
        
        rel_x = sx if sx < mx else (gx - 1 - sx)
        rel_y = sy if sy < my else (gy - 1 - sy)
        
        c0 = get_rid(rel_x, rel_y, gx)
        c1 = get_rid(gx - 1 - rel_x, rel_y, gx)
        c2 = get_rid(rel_x, gy - 1 - rel_y, gx)
        c3 = get_rid(gx - 1 - rel_x, gy - 1 - rel_y, gx)
        
        candidates = [c0, c1, c2, c3]
        if all(c in valid_cores_set for c in candidates):
            return sorted(list(set(candidates)))
        return []

    def find_nearest_in_pool(target_rid, needed, pool, occupied, gx):
        candidates = sorted(pool, key=lambda r: dist_manhattan(target_rid, r, gx))
        valid = [c for c in candidates if c not in occupied]
        return valid[:needed]

    def get_factors_pair(n):
        """Split n into two factors (a, b) where a*b=n, close to sqrt(n)."""
        if n <= 1: return (1, 1)
        a = int(math.sqrt(n))
        while n % a != 0:
            a -= 1
        b = n // a
        return sorted((a, b), reverse=True)

    # --- [1] Analysis Functions ---
    def parse_hardware(iter_dir):
        if iter_dir and iter_dir.parent.name == "results":
            run_d = iter_dir.parent.parent
        else:
            run_d = Path.cwd()
            
        hw_f = run_d / "hardware.json"
        st3_d = run_d / "stage_3_layer_grouping"
        
        v_cores, v_drams = [], []
        gx, gy = 8, 6 
        
        try:
            with open(hw_f, 'r') as f:
                d = json.load(f)
                grid_cfg = d.get("grid_config", {})
                gx = grid_cfg.get("grid_x", 8)
                gy = grid_cfg.get("grid_y", 6)
                cmps = sorted(d.get("component_mapping", []), key=lambda x: x["router_id"])
                for c in cmps:
                    if c["type"] == "core": v_cores.append(c["router_id"])
                    elif c["type"] == "dram": v_drams.append(c["router_id"])
        except: pass
        
        if not v_cores: v_cores = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
        return v_cores, v_drams, gx, gy, st3_d

    def analyze_workload(log_path, stage3_path, gid, nodes_list):
        flow_f = log_path / "flowID.csv"
        alloc_f = log_path / "allocation.csv"
        n_amt, t_amt, n_mac = {}, {}, {}
        
        # 1. Allocation (Real Data Analysis)
        if alloc_f.exists():
            try:
                import pandas as pd
                adf = pd.read_csv(alloc_f)
                # Compute MAC count (Identify Heavy Nodes)
                if 'compute_node' in adf.columns and 'compute_mac_count' in adf.columns:
                    macs = adf.groupby('compute_node')['compute_mac_count'].sum().to_dict()
                    for n, m in macs.items():
                        if n in nodes_list: n_mac[n] = n_mac.get(n, 0) + m
                # Compute Data Traffic (Identify Communication Intensity)
                if 'compute_node' in adf.columns and 'total_load_bytes' in adf.columns:
                        adf['total_bytes'] = adf['total_load_bytes'] + adf['total_store_bytes']
                        bytes_sum = adf.groupby('compute_node')['total_bytes'].sum().to_dict()
                        for n, b in bytes_sum.items():
                            if n in nodes_list: n_amt[n] = n_amt.get(n, 0) + b
            except: pass

        # 2. FlowID (Static Tensor Info)
        if flow_f.exists():
            try:
                import pandas as pd
                df = pd.read_csv(flow_f)
                if 'tensor' in df.columns and 'amount' in df.columns:
                    tsums = df.groupby('tensor')['amount'].sum().to_dict()
                    t_amt = tsums
                    if not n_amt: # Fallback if allocation.csv is missing
                        for t, a in tsums.items():
                            if "weight" in t: continue 
                            for n in nodes_list:
                                if n in t: n_amt[n] = n_amt.get(n, 0) + a
            except: pass

        # Default values to prevent division by zero
        for n in nodes_list:
            if n not in n_amt: n_amt[n] = 1.0
            if n not in n_mac: n_mac[n] = 1.0 

        # 3. Graph Construction (Bidirectional)
        lg_f = stage3_path / f"layer_group_{gid}.json"
        
        in_map = {n: [] for n in nodes_list}       # Parents
        out_map = {n: [] for n in nodes_list}      # Children
        parent_traf = {n: {} for n in nodes_list}  # Traffic from Parent
        child_traf = {n: {} for n in nodes_list}   # Traffic to Child
        n_cons, n_ein = {}, {}
        
        if lg_f.exists():
            try:
                with open(lg_f, 'r') as f:
                    lg = json.load(f)
                    n_info = lg.get("nodes", {})
                    e_info = lg.get("edges", {})
                    
                    # 3.1 Node Constraints & Einsum
                    for n, info in n_info.items():
                        if n in nodes_list:
                            n_cons[n] = info.get("partition_constraints", {})
                            eq = info.get("einsum_equation", "")
                            if "->" in eq: n_ein[n] = eq.split("->")[1].strip()
                    
                    # 3.2 Edge & Traffic Map
                    for edge in e_info.values():
                        src, dst = edge.get("source_node"), edge.get("target_node")
                        
                        w = n_amt.get(src, 1.0)
                        
                        # [Parent Mapping] dst's parent is src
                        if dst in nodes_list:
                            prod = n_info.get(src, {}).get("producer_node", src)
                            if prod not in in_map[dst]:
                                in_map[dst].append(prod)
                            parent_traf[dst][prod] = parent_traf[dst].get(prod, 0) + w
                            
                        # [Child Mapping] src's child is dst
                        if src in nodes_list and dst in nodes_list:
                            if dst not in out_map[src]:
                                out_map[src].append(dst)
                            child_traf[src][dst] = child_traf[src].get(dst, 0) + w
                                
            except: pass
            
        return n_amt, n_mac, t_amt, in_map, out_map, parent_traf, child_traf, n_cons, n_ein


    # --- [2] Allocation Logic (With Capacity Check) ---
    def calc_symmetric_allocation(c_nodes, n_amt, n_mac, n_cons, total_cores):
        counts = {}
        total_traffic = sum(n_amt.values()) or 1.0
        total_mac = sum(n_mac.values()) or 1.0
        
        avg_mac = total_mac / (len(n_mac) or 1)
        
        node_scores = {}
        for n in c_nodes:
            s = (n_amt[n] / total_traffic) * 0.4 + (n_mac[n] / total_mac) * 0.6
            node_scores[n] = s
        
        total_s = sum(node_scores.values()) or 1.0
        sorted_ns = sorted(c_nodes.keys(), key=lambda n: node_scores[n], reverse=True)
        
        # [1] Hardware Adaptive Limit (Total / 3, aligned to 4)
        raw_limit = total_cores // 3
        hw_limit = max(4, (raw_limit // 4) * 4) 
        
        # --- Initial Allocation ---
        for n in sorted_ns:
            raw = (node_scores[n] / total_s) * total_cores
            
            # Dynamic Boosting
            if n_mac[n] > avg_mac * 2.0: 
                raw *= 4.0 
            elif n_mac[n] > avg_mac * 0.8: 
                raw *= 2.0     
            
            # Universal Quantization
            cnt = round(raw / 4) * 4
            
            # Constraint Check
            cons = n_cons.get(n, {})
            max_limit = hw_limit 
            for k, v in cons.items():
                if isinstance(v, str) and 'aligned' in v:
                    try:
                        val = int(v.split(':')[1])
                        limit = max(1, 512 // val)
                        max_limit = min(max_limit, limit)
                    except: pass
            
            if cnt > max_limit: cnt = max_limit
            if cnt < 4: cnt = 4 
            
            counts[n] = int(cnt)
            
        # --- [CRITICAL FIX] Capacity Normalization ---
        # If allocated total exceeds hardware cores, reduce from low-priority nodes
        while sum(counts.values()) > total_cores:
            reduced = False
            # Iterate from lowest priority (end of sorted_ns) to highest
            for n in reversed(sorted_ns):
                if counts[n] > 4: # Can reduce only if it has > 4 cores
                    counts[n] -= 4
                    reduced = True
                    if sum(counts.values()) <= total_cores:
                        break
            
            # If we loop through everyone and can't reduce anymore (everyone is at 4), break.
            # This means Total Cores < 4 * Node Count (Hardware is too small for this graph)
            if not reduced:
                break
        # --- Phase 3: Utilization Boost (Distribute Leftovers) ---
        # If total allocated < total_cores, distribute remaining cores evenly(이븐하게)
        remaining = total_cores - sum(counts.values())
        
        if remaining > 0:
            # Loop until no remaining cores or no node can take more
            while remaining >= 4: # 4개 이상 남았을 때만
                distributed_this_round = False
                for n in sorted_ns: 
                    if remaining < 4: break # 4개 미만이면 중단
                    
                    # Check absolute constraints for this node
                    cons = n_cons.get(n, {})
                    node_max = total_cores # Relaxed limit for leftovers
                    for k, v in cons.items():
                        if isinstance(v, str) and 'aligned' in v:
                            try:
                                val = int(v.split(':')[1])
                                limit = max(1, 512 // val)
                                node_max = min(node_max, limit)
                            except: pass
                    
                    # If node can take more core
                    if counts[n] + 4 <= node_max:
                        counts[n] += 4  # <=== 4개씩 추가!
                        remaining -= 4
                        distributed_this_round = True
            
                if not distributed_this_round: break
                
        return counts, sorted_ns

        # --- [3] Dynamic Priority Placement (Fixed Fallback & Flexible Strategies) ---
    def execute_priority_placement(c_nodes, t_nodes, sorted_ns, counts, in_map, out_map, parent_traf, child_traf, t_amt, n_mac, n_cons, n_ein, v_cores, v_drams, gx, gy, iteration):
        occupied = set()
        v_cores_set = set(v_cores)
        
        # [SEED] Deterministic Randomness
        random.seed(iteration)
        
        # Resources (TL Quadrant)
        tl_cores = get_quadrant_pools(v_cores, gx, gy)[0]
        tl_rows = sorted(list(set(get_coords(c, gx)[1] for c in tl_cores)))
        min_y, max_y = min(tl_rows), max(tl_rows)
        
        # Helper: Get cores in specific TL row
        def get_tl_row_cores(r_y):
            return [c for c in tl_cores if get_coords(c, gx)[1] == r_y]

        # Categorize Nodes
        total_mac = sum(n_mac.values()) or 1.0
        avg_mac = total_mac / (len(n_mac) or 1)
        
        node_out_degree = {n: 0 for n in sorted_ns}
        for n in sorted_ns:
            parents = in_map.get(n, [])
            for p in parents:
                if p in node_out_degree: node_out_degree[p] += 1
        
        isolated_nodes = []
        heavy_nodes = []
        light_nodes = []
        
        for n in sorted_ns:
            parents = in_map.get(n, [])
            children = out_map.get(n, [])
            
            has_compute_parent = any(p in c_nodes for p in parents)
            has_compute_child = any(c in c_nodes for c in children)
            
            # Isolation: No compute dependency
            if not has_compute_parent and not has_compute_child:
                isolated_nodes.append(n)
            # Heavy: High MAC or High Count
            elif n_mac[n] > avg_mac or counts[n] >= 8:
                heavy_nodes.append(n)
            else:
                light_nodes.append(n)
        
        placement_order = isolated_nodes + heavy_nodes + light_nodes
        node_placement = {} # Global Placement Record

        # --- 1. Compute Node Placement ---
        for node in placement_order:
            needed = counts[node] # Global count
            per_quad = needed // 4
            if per_quad < 1: per_quad = 1
            
            chosen_tl = []
            
            # [STRATEGY A] Isolated Node (v_proj) -> Flexible Top-Down Fill
            # Modified: Try Row 0 first. If full, try Row 1, Row 2... (Prevents crash in dense layers)
            if node in isolated_nodes:
                for r_y in range(min_y, max_y + 1):
                    row_candidates = [c for c in get_tl_row_cores(r_y) if c not in occupied]
                    if len(row_candidates) >= per_quad:
                        # Sort Right -> Left (Max X -> Min X) for Center-Out mirroring
                        row_candidates.sort(key=lambda c: get_coords(c, gx)[0], reverse=True)
                        chosen_tl = row_candidates[:per_quad]
                        break
                # If strictly single row logic failed to find enough contiguous space,
                # Fallback to general candidates in TL (will be handled by fallback logic below if empty)
                if not chosen_tl:
                    fallback_candidates = [c for c in tl_cores if c not in occupied]
                    fallback_candidates.sort(key=lambda c: get_coords(c, gx)[0], reverse=True)
                    chosen_tl = fallback_candidates[:per_quad]
                
            # [STRATEGY B] Heavy Node (MatMul) -> Col 0 First, Top-Down Fill
            elif node in heavy_nodes:
                candidates = [c for c in tl_cores if c not in occupied]
                # Sort: Column (Asc) -> Row (Asc)
                # Fills Col 0 (Top->Bottom), then Col 1... strictly.
                candidates.sort(key=lambda c: (get_coords(c, gx)[0], get_coords(c, gx)[1]))
                chosen_tl = candidates[:per_quad]
                        
            # [STRATEGY C] Light Node -> Relaxed Gravity + Top-K Stochastic Sampling
            else:
                # 1. Calculate Gravity Center (기존 로직 유지)
                tx, ty, total_w = 0, 0, 0
                
                # Pull from Parents
                parents = in_map.get(node, [])
                for p in parents:
                    if p in node_placement:
                        w = parent_traf[node].get(p, 1.0)
                        for r in node_placement[p]:
                            rx, ry = get_coords(r, gx)
                            tx += rx * w; ty += ry * w; total_w += w
                            
                # Pull from Children
                children = out_map.get(node, [])
                for c in children:
                    if c in node_placement:
                        w = child_traf[node].get(c, 1.0) * 1.5 
                        for r in node_placement[c]:
                            rx, ry = get_coords(r, gx)
                            tx += rx * w; ty += ry * w; total_w += w
                
                if total_w > 0: 
                    target_x, target_y = round(tx/total_w), round(ty/total_w)
                else: 
                    target_x, target_y = gx//2, gy//2 
                
                # Map global target back to TL quadrant
                tl_target_rid = get_rid(min(max(target_x, 0), gx//2-1), min(max(target_y, 0), gy//2-1), gx)
                t_cx, t_cy = get_coords(tl_target_rid, gx)
                
                all_free = [c for c in tl_cores if c not in occupied]
                
                # 2. [수정됨] Strategy 2: Weight Relaxation (가중치 완화)
                # X축 가중치를 50 -> 8로 대폭 낮추어 옆줄(Column) 탐색 허용
                # Random Noise를 섞어 점수가 비슷하면 순위가 뒤바뀌게 함
                candidates = sorted(all_free, key=lambda c: (
                    abs(get_coords(c, gx)[0] - t_cx) * 8.0   # 50 -> 8.0 (수직 강제성 완화)
                    + abs(get_coords(c, gx)[1] - t_cy) * 1.0 # Y축 거리는 그대로
                    + random.uniform(0, 5.0)                 # 노이즈 추가
                ))
                
                # 3. [추가됨] Strategy 3: Top-K Stochastic Sampling (상위 후보군 내 랜덤 선택)
                # 필요한 개수(per_quad)보다 2배수 많은 후보를 뽑은 뒤, 그 안에서 랜덤 추첨
                # 예: 1개가 필요하면 상위 2~3개 중 하나를 뽑음 -> Local Minima 탈출
                pool_size = min(len(candidates), max(per_quad * 2, 4)) 
                
                if pool_size >= per_quad:
                    # 상위 pool_size개 중에서 per_quad개를 랜덤 비복원 추출
                    top_k_candidates = candidates[:pool_size]
                    chosen_tl = random.sample(top_k_candidates, per_quad)
                else:
                    # 후보가 부족하면 그냥 다 가져감
                    chosen_tl = candidates[:per_quad]
            
            # --- Symmetry Expansion ---
            assigned_global = []
            for seed in chosen_tl:
                sym_group = get_symmetric_cores(seed, gx, gy, v_cores_set)
                # Strict symmetry check
                if len(sym_group) == 4 and all(c not in occupied for c in sym_group):
                    assigned_global.extend(sym_group)
                else:
                    # If symmetry is broken (rare in TL logic, but possible in dense maps)
                    # Don't add partial group, just continue. Fallback will handle it.
                    pass 
            
            # [FIXED & CRITICAL] Robust Fallback Logic
            # If strategy failed or symmetry failed, find ANY available spot.
            # This prevents the "mapped to router 1" crash.
            if len(assigned_global) < needed:
                # 1. Fill from remaining needs
                rem_needed = needed - len(assigned_global)
                
                # Find strictly unoccupied cores across the WHOLE chip
                global_free = [c for c in v_cores if c not in occupied and c not in assigned_global]
                
                # Sort by proximity to center (heuristic) or just order
                # Sorting helps keep some locality even in fallback
                global_free.sort(key=lambda c: dist_manhattan(c, get_rid(gx//2, gy//2, gx), gx))
                
                assigned_global.extend(global_free[:rem_needed])
                
                # 2. Panic Mode (Should virtually never happen unless chip is 100% full)
                if not assigned_global:
                    assigned_global = [v_cores[0]] # Only then overwrite 0

            # Truncate to exact need
            assigned_global = assigned_global[:needed]

            # Register & Sort Spatially (Y then X for Tensor Mapping)
            occupied.update(assigned_global)
            final_rids = sorted(assigned_global, key=lambda r: (get_coords(r, gx)[1], get_coords(r, gx)[0]))
            
            c_nodes[node]["router_ids"] = final_rids
            node_placement[node] = final_rids
            
            # --- Partitioning (Factorization) ---
            dims = c_nodes[node].get("partition_dims", [1])
            new_dims = [1] * len(dims)
            safe_cnt = len(final_rids)
            
            out_idx = n_ein.get(node, "")
            cons = n_cons.get(node, {})
            split_candidates = []
            if len(out_idx) == len(dims):
                for i, char in enumerate(out_idx):
                    c_val = cons.get(char, "")
                    if c_val != "no_split": split_candidates.append(i)
            else:
                split_candidates = list(range(len(dims)))

            if safe_cnt > 1:
                factors = get_factors_pair(safe_cnt)
                if len(split_candidates) >= 2 and factors[1] > 1:
                        new_dims[split_candidates[0]] = factors[0]
                        new_dims[split_candidates[1]] = factors[1]
                elif split_candidates:
                        new_dims[split_candidates[0]] = safe_cnt
                else:
                        new_dims[0] = safe_cnt
            c_nodes[node]["partition_dims"] = new_dims

        # --- 2. Tensor Placement (Row Aligned + No Cap) ---
        drams_by_row = {y: [] for y in range(gy)}
        for d in v_drams:
            _, y = get_coords(d, gx)
            drams_by_row[y].append(d)
        
        # Global Inputs
        input_tensors = [t for t in t_nodes if "x" in t or "input" in t]
        for t_name in input_tensors:
            t_nodes[t_name]["router_ids"] = sorted(v_drams)
            dims = t_nodes[t_name].get("partition_dims", [1])
            new_dims = [1] * len(dims)
            if len(v_drams) > 1: new_dims[1 if len(dims)>=3 else 0] = len(v_drams)
            t_nodes[t_name]["partition_dims"] = new_dims
            
        # Intermediate Tensors
        remaining_tensors = [t for t in t_nodes if t not in input_tensors and "weight" not in t]
        
        for t_name in remaining_tensors:
            t_conf = t_nodes[t_name]
            producer = None
            for node in c_nodes:
                if node in t_name: producer = node; break
            
            assigned_drams = []
            if producer and producer in node_placement:
                prod_cores = node_placement[producer]
                prod_rows = sorted(list(set(get_coords(c, gx)[1] for c in prod_cores)))
                for y in prod_rows:
                    assigned_drams.extend(drams_by_row[y])
            
            if not assigned_drams:
                per_quad = 2 if t_amt.get(t_name,0) > 100000 else 1
                q_drams = get_quadrant_pools(v_drams, gx, gy)
                for q in range(4):
                    assigned_drams.extend(q_drams[q][:per_quad])
            
            unique_drams = sorted(list(set(assigned_drams)), key=lambda r: (get_coords(r, gx)[1], get_coords(r, gx)[0]))
            t_conf["router_ids"] = unique_drams
            
            dims = t_conf.get("partition_dims", [1])
            new_dims = [1] * len(dims)
            cnt = len(unique_drams)
            
            if cnt > 1:
                factors = get_factors_pair(cnt)
                limit_dim1 = 8
                
                if factors[0] <= limit_dim1:
                    a, b = factors
                else:
                    a, b = factors

                idx1 = 1 if len(dims) >= 3 else 0
                idx2 = 2 if len(dims) >= 3 else 1
                
                if idx2 < len(dims):
                    new_dims[idx1] = a
                    new_dims[idx2] = b
                else:
                    new_dims[idx1] = cnt
                    
            t_conf["partition_dims"] = new_dims

    # --- [4] Main Execution Flow ---
    valid_core_sorted, valid_dram_sorted, gx, gy, stage3_dir = parse_hardware(last_iter_dir)
    total_cores = len(valid_core_sorted)
    groups = allowed_groups or []

    # Iteration 1
    if iteration == 1:
        for gid in groups:
            key = f"layer_group_{gid}"
            if key not in partitions: continue
            c_nodes = partitions[key].get("compute_nodes", {})
            stride = max(1, total_cores // len(c_nodes)) if c_nodes else 1
            for i, (n, conf) in enumerate(c_nodes.items()):
                conf["router_ids"] = [valid_core_sorted[(i * stride) % total_cores]]
                conf["partition_dims"] = [1] * len(conf.get("partition_dims", [1]))

    # Iteration > 1
    else:
        for gid in groups:
            key = f"layer_group_{gid}"
            if key not in partitions: continue
            c_nodes = partitions[key].get("compute_nodes", {})
            t_nodes = partitions[key].get("tensor_nodes", {})
            if not c_nodes: continue
            
            c_list = list(c_nodes.keys())
            t_list = list(t_nodes.keys())
            full_list = c_list + t_list
            
            log_d = last_iter_dir / "logs" / key
            
            n_amt, n_mac, t_amt, in_map, out_map, p_traf, c_traf, n_cons, n_ein = analyze_workload(log_d, stage3_dir, gid, full_list)
            cnts, sorted_ns = calc_symmetric_allocation(c_nodes, n_amt, n_mac, n_cons, total_cores)
            execute_priority_placement(c_nodes, t_nodes, sorted_ns, cnts, in_map, out_map, p_traf, c_traf, t_amt, n_mac, n_cons, n_ein, valid_core_sorted, valid_dram_sorted, gx, gy, iteration)

    """ Edit code above """

    # Ensure mutated partitions stay attached
    config["partitions"] = partitions

    # Record a breadcrumb so users can see what changed
    meta = config.setdefault("_simple_dse", {})
    history = meta.get("history", [])
    history.append(
        {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "target_groups": allowed_groups if allowed_groups is not None else "all",
        }
    )
    meta["last_iteration"] = iteration

    return config


def _parse_layer_groups(layer_groups_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated layer group string into a list of ints."""
    if not layer_groups_str:
        return None
    groups: List[str] = []
    for token in layer_groups_str.split(","):
        token = token.strip()
        if not token:
            continue
        groups.append(token)
    return groups


def _list_available_groups(stage3_dir: Path) -> List[str]:
    """Discover layer group IDs from layer_group_*.json files."""
    ids: List[str] = []
    for path in stage3_dir.glob("layer_group_*.json"):
        try:
            ids.append(path.stem.split("_")[-1])
        except ValueError:
            continue
    return sorted(ids)


def _summarize_changes(
    before: Dict[str, Any],
    after: Dict[str, Any],
    allowed_groups: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract a concise list of changes between two configs (router_ids / partition_dims).
    Only inspects compute_nodes and tensor_nodes for target groups.
    """
    changes: List[Dict[str, Any]] = []

    before_partitions = before.get("partitions", {})
    after_partitions = after.get("partitions", {})

    for group_key, after_group in after_partitions.items():
        group_idx = (group_key.split("_")[-1]) if "_" in group_key else None
        if allowed_groups is not None and group_idx not in allowed_groups:
            continue

        before_group = before_partitions.get(group_key, {})

        for node_type in ("compute_nodes", "tensor_nodes"):
            before_nodes = before_group.get(node_type, {}) or {}
            after_nodes = after_group.get(node_type, {}) or {}

            for node_name, after_node in after_nodes.items():
                before_node = before_nodes.get(node_name, {})
                before_router = before_node.get("router_ids")
                after_router = after_node.get("router_ids")
                before_dims = before_node.get("partition_dims")
                after_dims = after_node.get("partition_dims")

                changed_fields: List[str] = []
                if before_router != after_router:
                    changed_fields.append("router_ids")
                if before_dims != after_dims:
                    changed_fields.append("partition_dims")

                if changed_fields:
                    entry: Dict[str, Any] = {
                        "group": group_key,
                        "node_type": node_type,
                        "node": node_name,
                        "changed_fields": changed_fields,
                    }
                    if "router_ids" in changed_fields:
                        entry["router_ids_before"] = before_router
                        entry["router_ids_after"] = after_router
                    if "partition_dims" in changed_fields:
                        entry["partition_dims_before"] = before_dims
                        entry["partition_dims_after"] = after_dims
                    changes.append(entry)

    return changes


def _build_env_with_systemc() -> Dict[str, str]:
    """Prepare environment with SystemC library on LD_LIBRARY_PATH."""
    env = os.environ.copy()
    systemc_lib = _find_systemc_lib()
    if systemc_lib:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{systemc_lib}:{existing}" if existing else str(systemc_lib)
        )
    return env


def _load_vis_template() -> Dict[str, Any]:
    """Load visualization template from TP_AS25 or return a minimal default."""
    try:
        return ConfigManager.load_config(VIS_TEMPLATE_PATH)
    except Exception:
        return {"plots": {}}


def _apply_usage_strings(vis_config: Dict[str, Any], workspace_path: Path) -> Dict[str, Any]:
    """Embed helpful usage strings pointing to the per-iteration workspace."""
    base_cmd = f"python scripts/generate_plots.py --workspace {workspace_path}"
    vis_config["_usage"] = base_cmd

    plots = vis_config.get("plots")
    if isinstance(plots, dict):
        for plot_name, entry in plots.items():
            if isinstance(entry, dict):
                entry["_usage"] = f"{base_cmd} --plot-type {plot_name}"

    return vis_config


def _run_tlm_sim(
    simulator: Path,
    group_file: Path,
    config_file: Path,
    hardware_file: Path,
    group_id: str,
    log_dir: Optional[Path],
    timeout_s: int = 300,
) -> Tuple[Dict[str, float], str, str]:
    """
    Run the TLM simulator binary directly and parse stdout for metrics.

    Returns (metrics, stdout, stderr).
    """
    if not simulator.exists():
        raise FileNotFoundError(f"TLM simulator not found at {simulator}")
    if not group_file.exists():
        raise FileNotFoundError(f"Group file not found: {group_file}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not hardware_file.exists():
        raise FileNotFoundError(f"Hardware file not found: {hardware_file}")

    cmd = [str(simulator), str(group_file), str(config_file), str(hardware_file), group_id]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd.append(f"--log-dir={log_dir}")

    env = _build_env_with_systemc()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=Path.cwd(),
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Simulation failed for {group_id} (rc={result.returncode}).\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    metrics = _parse_simulator_output(result.stdout, group_id)
    metrics["success"] = True
    return metrics, result.stdout, result.stderr


def _parse_simulator_output(output: str, group_id: str) -> Dict[str, float]:
    """Parse simulator stdout for metrics."""
    metrics: Dict[str, Optional[float]] = {
        "execution_time": None,
        "energy": None,
        "area": None,
        "cost": None,
    }

    for line in output.split("\n"):
        line = line.strip()
        if "simulation time:" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                value_str = parts[1].strip().split()[0]
                metrics["execution_time"] = float(value_str)
        elif "throughput:" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                value_str = parts[1].strip().split()[0]
                metrics["throughput"] = float(value_str)
        elif "energy consumption:" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                value_str = parts[1].strip().split()[0]
                metrics["energy"] = float(value_str)
        elif "total area:" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                value_str = parts[1].strip().split()[0]
                metrics["area"] = float(value_str)
        elif "monetary cost:" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                value_str = parts[1].strip().split()[0]
                metrics["cost"] = float(value_str)
        elif "power efficiency:" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                value_str = parts[1].strip().split()[0]
                metrics["power_effs"] = float(value_str)

    missing = [k for k, v in metrics.items() if v is None]
    if missing:
        raise RuntimeError(
            f"Missing required metrics for {group_id} from simulator output. Missing: {missing}\n"
            f"Output:\n{output}"
        )

    return metrics  # type: ignore[return-value]


def _aggregate(group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Aggregate per-group metrics into summary and group detail list."""
    execution_times = []
    energies = []
    throughputs = []
    power_effs = []
    areas = []
    costs = []
    group_details = []

    for group_name, metrics in sorted(group_metrics.items()):
        group_idx = group_name.split("_")[-1]
        execution_times.append(metrics["execution_time"])
        energies.append(metrics["energy"])
        areas.append(metrics["area"])
        costs.append(metrics["cost"])
        throughputs.append(metrics["throughput"])
        power_effs.append(metrics["power_effs"])
        
        group_details.append(
            {
                "group_id": group_idx,
                "status": "completed",
                "simulation_time_ns": metrics["execution_time"],
                "energy_mj": metrics["energy"],
                "throughput_ops": metrics["throughput"],
                "power_efficiency_ops_per_w": metrics["power_effs"],
                "area_mm2": metrics["area"],
                "cost_usd": metrics.get("cost", 0.0),
            }
        )

    total_energy = sum(energies)
    total_throughput = sum(throughputs) / len(throughputs)
    total_power_effs = sum(power_effs) / len(throughputs)

    return {
        "summary": {
            "total_groups": len(group_metrics),
            "total_simulation_time_ns": sum(execution_times),
            "total_energy_mj": total_energy,
            "total_throughput_ops": total_throughput,
            "power_efficiency_ops_per_w": total_power_effs,
            "total_area_mm2": max(areas) if areas else 0.0,
            "max_cost_usd": max(costs) if costs else 0.0,
        },
        "groups": group_details,
    }


def _prepare_iteration_vis_config(
    iteration: int,
    iter_dir: Path,
    iter_config_path: Path,
    iter_result_file: Path,
    iter_logs_dir: Path,
) -> Path:
    """
    Prepare visualization artifacts directly in the iteration directory.
    Returns path to the generated vis_report_config.json.
    """
    # Ensure logs are present
    logs_dir = iter_logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build vis_report_config.json based on TP_AS25 template (placed under iter_dir)
    vis_config = _load_vis_template()
    vis_config = _apply_usage_strings(vis_config, iter_dir)
    vis_config_path = iter_dir / "vis_report_config.json"
    ConfigManager.save_config(vis_config_path, vis_config)

    # logging.info(
    #     "Prepared visualization config for iteration %s at %s (logs -> %s)",
    #     iteration,
    #     vis_config_path,
    #     logs_dir,
    # )

    return vis_config_path


def _copy_workspace_inputs(manual_ws: Path, run_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Copy required files from manual workspace into run_dir.
    Returns (stage3_dir, stage4_config_path, hardware_path) in the run_dir.
    """
    stage3_src = manual_ws / "stage_3_layer_grouping"
    stage4_src = manual_ws / "stage_4_simulation" / "config.json"
    hardware_src = manual_ws / "hardware.json"

    if not stage3_src.exists():
        raise FileNotFoundError(f"Stage 3 directory missing: {stage3_src}")
    if not stage4_src.exists():
        raise FileNotFoundError(f"Stage 4 config missing: {stage4_src}")
    if not hardware_src.exists():
        raise FileNotFoundError(f"hardware.json missing: {hardware_src}")

    stage3_dst = run_dir / "stage_3_layer_grouping"
    stage4_dst_dir = run_dir / "stage_4_simulation"
    stage4_dst = stage4_dst_dir / "config.json"
    hardware_dst = run_dir / "hardware.json"

    stage3_dst.mkdir(parents=True, exist_ok=True)
    stage4_dst_dir.mkdir(parents=True, exist_ok=True)

    for file in stage3_src.glob("layer_group_*.json"):
        shutil.copy(file, stage3_dst / file.name)

    shutil.copy(stage4_src, stage4_dst)
    shutil.copy(hardware_src, hardware_dst)

    logging.info("Copied inputs into run dir: %s", run_dir)
    logging.info("  Stage3 groups -> %s", stage3_dst)
    logging.info("  Stage4 config -> %s", stage4_dst)
    logging.info("  Hardware      -> %s", hardware_dst)

    return stage3_dst, stage4_dst, hardware_dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal DSE loop on a manual workspace.")
    parser.add_argument("workspace", help="Name of the manual workspace to use.")
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Number of DSE iterations.",
    )
    parser.add_argument(
        "--layer-groups",
        default=None,
        help="Comma-separated list of layer group IDs to simulate (e.g., 0,2). Default: all groups.",
    )
    parser.add_argument(
        "--output-root",
        default="workspaces/dse",
        help="Root directory to store DSE logs (default: workspaces/dse).",
    )

    args = parser.parse_args()

    if args.iterations <= 0:
        raise SystemExit("iterations must be a positive integer")

    layer_groups = _parse_layer_groups(args.layer_groups)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] %(message)s", datefmt="%H:%M:%S",)

    workspace_manager = WorkspaceManager()
    manual_workspace = workspace_manager.get_workspace_path(args.workspace)

    if not manual_workspace.exists():
        raise SystemExit(f"Workspace not found: {manual_workspace}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{args.workspace}_{timestamp}"
    results_dir = run_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    stage3_dir, config_path, hardware_path = _copy_workspace_inputs(manual_workspace, run_dir)
    available_groups = _list_available_groups(stage3_dir)
    if not available_groups:
        raise SystemExit(f"No layer_group_*.json files found in {stage3_dir}")

    if layer_groups is None:
        target_groups = available_groups
    else:
        missing = sorted(set(layer_groups) - set(available_groups))
        if missing:
            raise SystemExit(f"Requested layer groups not found: {missing}")
        target_groups = sorted(layer_groups)

    simulator_path = Path(SIMULATOR_PATH)

    run_log: list[Dict[str, Any]] = []
    meta = {
        "workspace": args.workspace,
        "manual_workspace_path": str(manual_workspace),
        "iterations": args.iterations,
        "layer_groups": target_groups,
        "sim_mode": "tlm",
        "started_at": timestamp,
        "output_dir": str(run_dir),
        "simulator": str(simulator_path),
    }
    ConfigManager.save_config(run_dir / "run_meta.json", meta)

    best_tracker: Dict[str, Any] = {
        "iteration": None,
        "summary": {},
        "config_file": None,
        "result_file": None,
        "summary_file": None,
        "config_changes_file": None,
        "vis_config_file": None,
    }

    for iteration in range(1, args.iterations + 1):
        logging.info("Iteration %s/%s", iteration, args.iterations)

        iter_dir = results_dir / f"iter_{iteration:03d}"
        last_iter_dir = results_dir / f"iter_{iteration-1:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_logs_dir = iter_dir / "logs"
        iter_logs_dir.mkdir(parents=True, exist_ok=True)

        config = ConfigManager.load_config(config_path)
        before_config = copy.deepcopy(config)
        cur_config = config_selector(config, iteration, allowed_groups=target_groups, last_iter_dir=last_iter_dir)
        ConfigManager.save_config(config_path, cur_config)
        logging.info("Modified config written back to %s", config_path)

        change_list = _summarize_changes(before_config, cur_config, allowed_groups=target_groups)

        iter_config_path = iter_dir / "config.json"
        ConfigManager.save_config(iter_config_path, cur_config)
        logging.info("Saved cur_config config for iteration %s to %s", iteration, iter_config_path)

        iter_changes_path = iter_dir / "config_changes.json"
        ConfigManager.save_config(
            iter_changes_path,
            {"iteration": iteration, "changes": change_list},
        )
        logging.info(
            "Recorded %s config change(s) for iteration %s at %s",
            len(change_list),
            iteration,
            iter_changes_path,
        )

        # Run simulator per target group
        group_metrics: Dict[str, Dict[str, float]] = {}
        for gid in target_groups:
            group_name = f"layer_group_{gid}"
            group_file = stage3_dir / f"{group_name}.json"
            group_log_dir = iter_logs_dir / group_name

            try:
                metrics, stdout, stderr = _run_tlm_sim(
                    simulator_path, group_file, config_path, hardware_path, group_name, group_log_dir
                )
            except Exception as exc:
                logging.error("Simulation failed for %s: %s", group_name, exc)
                raise

            group_metrics[group_name] = metrics

            # Save raw outputs alongside iteration root for easy inspection
            # (iter_dir / f"{group_name}_stdout.txt").write_text(stdout)
            # if stderr.strip():
            #     (iter_dir / f"{group_name}_stderr.txt").write_text(stderr)
            logging.info(
                "Completed simulation for %s (iter %s): time_ns=%s energy_mj=%s area_mm2=%s cost_usd=%s",
                group_name,
                iteration,
                metrics.get("execution_time", "n/a"),
                metrics.get("energy", "n/a"),
                metrics.get("area", "n/a"),
                metrics.get("cost", "n/a"),
            )

        # Aggregate metrics and save stage-like result files
        aggregated = _aggregate(group_metrics)
        simulation_result = {
            "success": True,
            "message": f"Simulated {len(target_groups)} groups via simple DSE",
            "data": aggregated,
            "timestamp": datetime.now().isoformat(),
        }

        iter_result_file = iter_dir / "simulation_result.json"
        ConfigManager.save_config(iter_result_file, simulation_result)

        # Prepare visualization config/workspace for this iteration
        vis_config_path = _prepare_iteration_vis_config(
            iteration, iter_dir, iter_config_path, iter_result_file, iter_logs_dir
        )

        run_log.append(
            {
                "iteration": iteration,
                "config_file": str(iter_config_path),
                "result_file": str(iter_result_file),
                "config_changes_file": str(iter_changes_path),
                "vis_config_file": str(vis_config_path),
                "summary": aggregated["summary"],
            }
        )

        summary = aggregated["summary"]
        current_time = summary.get("total_simulation_time_ns")
        if isinstance(current_time, (int, float)):
            prev_summary = best_tracker.get("summary") or {}
            best_time = prev_summary.get("total_simulation_time_ns")
            if best_time is None or current_time < best_time:
                best_tracker = {
                    "iteration": iteration,
                    "summary": summary,
                    "config_file": str(iter_config_path),
                    "result_file": str(iter_result_file),
                    "config_changes_file": str(iter_changes_path),
                    "vis_config_file": str(vis_config_path),
                }

        logging.info(
            "Finished run %s: total_simulation_time_ns=%s, energy_mj=%s, area_mm2=%s, cost_usd=%s, config_changes=%s",
            iteration,
            summary.get("total_simulation_time_ns", "n/a"),
            summary.get("total_energy_mj", "n/a"),
            summary.get("total_area_mm2", "n/a"),
            summary.get("max_cost_usd", "n/a"),
            len(change_list),
        )
        logging.info("Iteration %s outputs saved under %s", iteration, iter_dir)

    ConfigManager.save_config(run_dir / "run_log.json", {"iterations": run_log})
    if best_tracker["iteration"] is not None:
        # Save overall summary/best at run_dir level (single file)
        overall_summary = {
            "best_iteration": best_tracker["iteration"],
            "summary": best_tracker.get("summary", {}),
            "config_file": best_tracker.get("config_file"),
            "result_file": best_tracker.get("result_file"),
            "config_changes_file": best_tracker.get("config_changes_file"),
            "vis_config_file": best_tracker.get("vis_config_file"),
        }
        ConfigManager.save_config(run_dir / "summary.json", overall_summary)
        logging.info(
            "Best result: iteration %s with total_simulation_time_ns=%s (details in %s)",
            best_tracker["iteration"],
            (best_tracker.get("summary") or {}).get("total_simulation_time_ns", "n/a"),
            run_dir / "summary.json",
        )
    else:
        logging.info("No valid summaries found to select a best result.")

    logging.info("All iterations complete. Logs saved to %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
