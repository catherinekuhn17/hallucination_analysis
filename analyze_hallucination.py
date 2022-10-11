import sys, glob, os, shutil
import numpy as np
import pandas as pd
import mdtraj as md
import glob
from Bio.PDB import *
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pymol2
from Bio.SVDSuperimposer import SVDSuperimposer
import warnings
warnings.filterwarnings("ignore")
import pickle
from argparse import ArgumentParser
from distutils.util import strtobool

def get_args(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=str, help="path to folder you want to run analysis on")
    p.add_argument("--h_info_from_trb", type=strtobool, default=True, help="pull histidine placement info from trb file")
    p.add_argument("--hist_place", type=str, help="positions of hist in hallucination")
    p.add_argument("--ref_fn", type=str, help="path to template pdb")
    p.add_argument("--hist_place_ref", type=str, help="positions of hist in template pdb")
    p.add_argument("--out_file", type=str, help="fn to write output to")
    args = p.parse_args()
    if argv is not None:
        args = p.parse_args(argv) # for use when testing
    else:
        args = p.parse_args()
    return args

def load_pdb_and_remove_H(fn, name, tmp_fn):
    '''
    we don't want to care about Hydrogens later so it's easiest to remove them 
    '''
    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(fn, name)
        pymol.cmd.remove('hydrogen')
        pymol.cmd.save(tmp_fn)
    p = PDBParser()
    s = p.get_structure(name, tmp_fn)
    return s

def build_af_score_dict(fn):
    df = pd.read_csv(fn)
    name = [d[len(d)-6:len(d)-4] for d in df['name']]
    print(name)
    #''.join([n for n in n2 if n.isdigit()])
    name =  [''.join([n for n in n2 if n.isdigit()]) for n2 in name]
    lddt={}
    rmsd={}
    for n, l, r in zip(name, df['af2_lddt'], df['rmsd_af2_des']):
        lddt[int(n)]=l
        rmsd[int(n)]=r
    return lddt, rmsd

def build_struct_dict(folder, h_info_from_trb, hist_place=None, hist_place_ref=None):
    out_dict={}
    fns_dict={}
    H_place_info={}
    ref_atoms={}
    fns = glob.glob(folder)
    for fn in fns:
        name = fn[len(fn)-10:len(fn)-8]
        name =  ''.join([n for n in name if n.isdigit()])
        tmp_fn = fn[0:-4]+'no_H.pdb'
        out_dict[int(name)]=load_pdb_and_remove_H(fn, name, tmp_fn)
        fns_dict[int(name)] = fn
        if 'af2' not in folder:
            trb_fn = fn[0:-8]+'.trb'
            file = open(trb_fn,'rb')
            object_file = pickle.load(file)
            if h_info_from_trb:
                atom_list=object_file['con_hal_pdb_idx']
                ref_atoms=object_file['con_ref_pdb_idx']
            else:
                atom_list = hist_place
                ref_atoms = hist_place_ref
            H_place_info[int(name)] = atom_list

    return out_dict, H_place_info, ref_atoms

def get_coords(struct_dict, res_list):
    '''
    Get xyz coords of all atoms for residues in res_list
    '''
    all_xyz_coord_dict={}
    for k, struct in struct_dict.items():
      #  print(k)
        all_xyz_coord = np.array([])
        res_list_tmp = res_list
        for chain, resi in res_list_tmp:
            xyz_coord=[]
            res_obj = struct[0][chain][int(resi)]
            atoms = list(res_obj.get_atoms())
            for a in atoms:
                xyz_coord.append(a.get_coord())
            all_xyz_coord=np.append(all_xyz_coord,xyz_coord)
        all_xyz_coord_dict[k]=all_xyz_coord.reshape(-1,3)
    return all_xyz_coord_dict

def get_rmsd(xyz_dict, ref_xyz_coords):
    rmsd_dict={}
    for k, xyz_coord in xyz_dict.items():
        sup = SVDSuperimposer()
        sup.set(ref_xyz_coords, xyz_coord)
        sup.run()
        rms = sup.get_rms()
        rmsd_dict[k] = rms
    return rmsd_dict

def get_rmsd_bb(ref_fn,  ref_H_place, struct_dict, res_list):
    xyz_coord=[]
    cb_coord=[]
    ref_struct = load_pdb_and_remove_H(ref_fn, 'tmp', 'tmp_noH.pdb')
    for chain, resi in ref_H_place:
        res_obj = ref_struct[0][chain][int(resi)]
        xyz_tmp=[res_obj['O'].get_coord(),
            res_obj['C'].get_coord(),
            res_obj['CA'].get_coord(),
            res_obj['N'].get_coord(),
            res_obj['CB'].get_coord()]
        xyz_coord.append(xyz_tmp)
        cb_coord.append(res_obj['CB'].get_coord())

    ref_xyz_coords=np.array(xyz_coord).flatten().reshape(-1,3)
    ref_cb_coord=cb_coord
    rmsd_dict={}
    cb_rmsd_dict={}
    bb_coords={}
    for k, struct in struct_dict.items():
        dih_angs=np.array([])
        res_list_tmp = res_list
        xyz_coord=[]
        cb_coord=[]
        for chain, resi in res_list_tmp:
            res_obj = struct[0][chain][int(resi)]
            cb = res_obj['CB'].get_coord()
            xyz_tmp=[res_obj['O'].get_coord(),
                    res_obj['C'].get_coord(),
                    res_obj['CA'].get_coord(),
                    res_obj['N'].get_coord(),
                    res_obj['CB'].get_coord()]
            xyz_coord.append(xyz_tmp)
            cb_coord.append(cb)
        xyz_coord = np.array(xyz_coord).flatten().reshape(-1,3)
        #for planar
        sup = SVDSuperimposer()
        sup.set(ref_xyz_coords, xyz_coord)
        sup.run()
        rms = sup.get_rms()
        rmsd_dict[k] = rms
        sup2 = SVDSuperimposer()
        sup2.set(np.array(ref_cb_coord), np.array(cb_coord))
        sup2.run()
        rms2 = sup2.get_rms()
        cb_rmsd_dict[k] = rms2    
        bb_coords[k]=xyz_coord
    return rmsd_dict, cb_rmsd_dict,  bb_coords

def main():
    '''
    This is now for loading in the PDBs of hallucinated structures (both the relaxed
    and the alphafold predicted ones)

    We are building it as a dict, with the format:
         {rlx : {structure # : structures},
         {af2 : {structure # : structures}}
    '''
    # arguments & settings
    args = get_args()
    print(args)
    hist_place=args.hist_place
    hist_place_ref=args.hist_place_ref
    folder=args.folder
    if hist_place is not None and hist_place_ref is not None:
        hist_place=np.array(args.hist_place.split(',')).reshape(-1,2)
        hist_place_ref=np.array(args.hist_place_ref.split(',')).reshape(-1,2)
    # initializing dictionaries
    pdb_structs={}
    file_names={}
    H_place_info={}
    lddt_score={}
    rmsd_af2={}
    ref_H_place_info={}
    
    lddt,rmsd = build_af_score_dict(f'{folder}/af2_metrics.csv') # get af metrics
    for types in ['rlx', 'af2']:
        if types == 'af2':
            struct, h_info, h_info_ref = build_struct_dict(f'{folder}/af2/*_rlx.pdb', 
                                                                args.h_info_from_trb,
                                                                hist_place,
                                                                hist_place_ref) # get af2 pdbs
        elif types == 'rlx':
            struct, h_info, h_info_ref = build_struct_dict(f'{folder}/*_rlx.pdb',
                                                                args.h_info_from_trb,
                                                                hist_place,
                                                                hist_place_ref) # get relaxed pdbs

        pdb_structs[types] = struct
        lddt_score[types] = lddt
        rmsd_af2[types] = rmsd    
        H_place_info[types] = h_info
        ref_H_place_info[types] = h_info_ref 

    # this is getting coordinate info of template so that we can get rmsd to template
    coords_dict={}
    dist_dict={}
    dih_dict={}
    for types in pdb_structs.keys():
        coords = get_coords(pdb_structs[types], 
                            H_place_info['rlx'][0])
        coords_dict[types]=coords

    # this is getting actual template info
    rmsd_dict_bb={}
    cb_rmsd_dict={}
    bb_coords={}
    for types in pdb_structs.keys():
        for types in pdb_structs.keys():      
            rmsd_dict_bb[types], cb_rmsd_dict[types], bb_coords[types]= get_rmsd_bb(args.ref_fn,
                                                  ref_H_place_info['rlx'],
                                                  pdb_structs[types], 
                                                  H_place_info['rlx'][0])


    rlx_or_af2=[]
    itter_nums=[]
    rmsd_bb=[]
    rmsd_af2_list=[]
    lddt_af2_list=[]
    cb_rmsd_dict_list=[]
    for types in pdb_structs.keys():
        for itter in pdb_structs['af2'].keys():
            rlx_or_af2.append(types)
            rmsd_bb.append(rmsd_dict_bb[types][itter])
            cb_rmsd_dict_list.append(cb_rmsd_dict[types][itter])
            itter_nums.append(itter)
            if 'af2' in rmsd_af2.keys():
                rmsd_af2_list.append(rmsd_af2['af2'][itter])
                lddt_af2_list.append(lddt_score['af2'][itter])
            elif 'rlx' in rmsd_af2.keys():
                rmsd_af2_list.append(rmsd_af2['rlx'][itter])
                lddt_af2_list.append(lddt_score['rlx'][itter])               

    df_all = pd.DataFrame()
    df_all['type'] = rlx_or_af2
    df_all['itter_nums'] = itter_nums
    df_all['rmsd_bb'] = rmsd_bb
    df_all['rmsd_cb'] = cb_rmsd_dict_list
    df_all['rmsd_af2'] = rmsd_af2_list
    df_all['lddt_af2'] = lddt_af2_list
    df_all.to_csv(f'{args.out_file}.csv')
    
if __name__ == "__main__":
    main()