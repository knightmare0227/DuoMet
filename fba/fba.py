import cobra
import pandas as pd
import numpy as np
from gapfilling import add_gapfilled_reactions
from optlang.interface import OPTIMAL

def flux_balance_analysis(model, paras, fraction_of_optimum=0.999999):
    # 1) 基础 FBA
    fba_solution = model.optimize()
    if model.solver.status != OPTIMAL:
        fba_solution = model.optimize()
        if model.solver.status != OPTIMAL:
            raise RuntimeError("FBA optimization failed with GLPK.")
    biomass = fba_solution.objective_value

    # 获取 biomass 反应（假设目标是单个反应）
    obj_vars = list(model.objective.variables)
    if len(obj_vars) == 0:
        raise RuntimeError("No objective variables found.")
    biomass_rxn_id = obj_vars[0].name
    biomass_rxn = model.reactions.get_by_id(biomass_rxn_id)
    original_biomass_lb = biomass_rxn.lower_bound

    # 施加 biomass 最低产量约束（通过修改下界）
    new_lb = max(original_biomass_lb, biomass * fraction_of_optimum)
    biomass_rxn.lower_bound = new_lb

    target_rxns = paras['TARGET_EX_RXNS']
    results = []

    for rxn_id in target_rxns:
        if rxn_id not in model.reactions:
            results.append({
                'reaction': rxn_id,
                'minimum': 0.0,
                'maximum': 0.0,
                'biomass': biomass,
                'normalized_maximum': 0.0,
                'phenotype': 0
            })
            continue

        rxn = model.reactions.get_by_id(rxn_id)

        # 最大化
        model.objective = rxn
        model.objective.direction = 'max'
        try:
            sol_max = model.optimize()
            max_flux = sol_max.fluxes[rxn_id] if sol_max.status == OPTIMAL else 0.0
        except Exception:
            max_flux = 0.0

        # 最小化
        model.objective.direction = 'min'
        try:
            sol_min = model.optimize()
            min_flux = sol_min.fluxes[rxn_id] if sol_min.status == OPTIMAL else 0.0
        except Exception:
            min_flux = 0.0

        norm_max = max_flux / biomass if biomass > 1e-8 else 0.0
        phenotype = 1 if norm_max >= float(paras['FLUX_CUTOFF']) else 0

        results.append({
            'reaction': rxn_id,
            'minimum': min_flux,
            'maximum': max_flux,
            'biomass': biomass,
            'normalized_maximum': norm_max,
            'phenotype': phenotype
        })

    # 恢复 biomass 反应的原始下界
    biomass_rxn.lower_bound = original_biomass_lb

    # 恢复原始目标
    model.objective = biomass_rxn
    model.objective.direction = 'max'

    fva = pd.DataFrame(results)
    # 确保 'reaction' 是普通列，不设置索引名
    # 如果之前有索引名，重置
    if fva.index.name is not None:
        fva = fva.reset_index(drop=True)
    return fva

def predict_fermentation(gem_file, universe, paras):
    print('predicting fermentation: %s...' % gem_file)
    model_no_gapfill, model_w_gapfill, num_rxns_added, rids_added = add_gapfilled_reactions(gem_file, universe, paras)

    fva_no_gapfill = flux_balance_analysis(model_no_gapfill, paras)
    fva_no_gapfill.columns = [c + '__no_gapfill' if c != 'reaction' else 'reaction' for c in fva_no_gapfill.columns]

    fva_w_gapfill = flux_balance_analysis(model_w_gapfill, paras)
    fva_w_gapfill.columns = [c + '__w_gapfill' if c != 'reaction' else 'reaction' for c in fva_w_gapfill.columns]

    fva = pd.merge(fva_no_gapfill, fva_w_gapfill, left_on='reaction', right_on='reaction', how='inner')
    fva['gem_file'] = gem_file.rstrip('.xml')
    fva['random_rxns'] = paras['ADD_RANDOM_RXNS']
    fva['num_rxns_to_add'] = paras['NUM_GAPFILLED_RXNS_TO_ADD']
    fva['num_rxns_added'] = num_rxns_added
    fva['rxn_ids_added'] = ';'.join(rids_added)
    fva['key_rxns'] = np.nan

    return fva