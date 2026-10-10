import xml.etree.ElementTree as ET
import csv
import random
import numpy as np
import torch

def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

def kernelToDistance(k):
    di = torch.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d

def getGipKernel(y, trans, gamma):
    if trans:
        y = y.T
    krnl = torch.mm(y, y.T)
    krnl = krnl / torch.mean(torch.diag(krnl))
    krnl = torch.exp(-kernelToDistance(krnl) * gamma)
    return krnl

def read_xml(xml_content):
    root = ET.fromstring(xml_content)
    ns = {
        'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
        'fbc': 'http://www.sbml.org/sbml/level3/version1/fbc/version2'
    }

    list_of_reactions = root.find('.//sbml:model/sbml:listOfReactions', ns)
    reactions = []
    for reaction in list_of_reactions.findall('sbml:reaction', ns):
        bigg_id = reaction.get('id').replace('R_', '')
        reactions.append(bigg_id)

    metabolites_set = set()
    reaction_metabolites = {}
    for reaction_elem in list_of_reactions.findall('sbml:reaction', ns):
        rxn_id = reaction_elem.get('id').replace('R_', '')
        metas = []
        for reactant in reaction_elem.findall('sbml:listOfReactants/sbml:speciesReference', ns):
            species_id = reactant.get('species').replace('M_', '')
            metabolites_set.add(species_id)
            metas.append(species_id)
        for product in reaction_elem.findall('sbml:listOfProducts/sbml:speciesReference', ns):
            species_id = product.get('species').replace('M_', '')
            metabolites_set.add(species_id)
            metas.append(species_id)
        reaction_metabolites[rxn_id] = list(dict.fromkeys(metas))

    metabolites = sorted(list(metabolites_set))

    with open('metabolite.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bigg_id', 'node_id'])
        for idx, meta in enumerate(metabolites):
            writer.writerow([meta, idx])

    with open('reaction.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bigg_id', 'hyperedge_id'])
        for idx, rxn in enumerate(reactions):
            writer.writerow([rxn, idx])

    meta_to_id = {meta: idx for idx, meta in enumerate(metabolites)}
    rxn_to_id = {rxn: idx for idx, rxn in enumerate(reactions)}

    with open('incidence_matrix_pos.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hyperedge_id', 'node_id'])
        for rxn, metas in reaction_metabolites.items():
            hyperedge_id = rxn_to_id[rxn]
            for meta in metas:
                node_id = meta_to_id[meta]
                writer.writerow([hyperedge_id, node_id])

    mmi_edges = set()
    for reaction_elem in list_of_reactions.findall('sbml:reaction', ns):
        rxn_id = reaction_elem.get('id').replace('R_', '')
        substrates = []
        products = []
        for reactant in reaction_elem.findall('sbml:listOfReactants/sbml:speciesReference', ns):
            species_id = reactant.get('species').replace('M_', '')
            substrates.append(species_id)
        for product in reaction_elem.findall('sbml:listOfProducts/sbml:speciesReference', ns):
            species_id = product.get('species').replace('M_', '')
            products.append(species_id)

        for sub in substrates:
            for prod in products:
                if sub != prod:
                    sub_id = meta_to_id.get(sub)
                    prod_id = meta_to_id.get(prod)
                    if sub_id is not None and prod_id is not None:
                        mmi_edges.add((sub_id, prod_id))

    with open('mmi.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['substrate_id', 'product_id'])
        for sub, prod in mmi_edges:
            writer.writerow([sub, prod])