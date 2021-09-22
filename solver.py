#
# Exercise scheduling with Mixed Integer Linear Programming
#
import numpy as np 
import json 
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def main(spec):

    G = create_group_matrix(spec)

    C = create_constraint_matrix(spec)
    
    model = make_model(G, C, spec)

    opt = SolverFactory('glpk')

    results = opt.solve(model) 

    #model.display()

    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print ("this is feasible and optimal")
    elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print ("infeasible solution")
        exit()
    else:
        print (str(results.solver))
        exit()

    D = np.zeros((G.shape[0], spec['n_days']))
    for eid, e in enumerate(G.index):
        for d in np.arange(spec['n_days']):
            D[eid, d] = pyo.value(model.v_exercise_schedule[e,d])

    ed = spec['exercise_durations']
    DM = np.array([ed[e]['sets'] * (ed[e]['set_duration_sec'] + ed[e]['rest_period_sec']) for e in G.index])
    
    D_time_sec = D * DM[:,np.newaxis]
    schedule_time_min = pd.DataFrame(data=D_time_sec / 60, index=G.index, columns=spec['day_names'])
    print(schedule_time_min)

    print(np.sum(schedule_time_min, axis=0))
def create_group_matrix(spec):

    exercises = sorted(spec['exercises'])
    eid = dict(zip(exercises, range(len(exercises))))

    groups = sorted(set(spec['exercises'].values()))
    gid = dict(zip(groups, range(len(groups))))

    G = np.zeros((len(exercises), len(groups)), dtype=int)
    for e, g in spec['exercises'].items():
        G[ eid[e], gid[g] ] = 1

    return pd.DataFrame(data=G, index=exercises, columns=groups)

def create_constraint_matrix(spec):

    exercises = sorted(spec['exercise_counts'])
    eid = dict(zip(exercises, range(len(exercises))))

    C = [spec['exercise_counts'][e] for e in exercises]

    return pd.DataFrame(data=C, index=exercises, columns=['count']).astype(int)

def make_model(G, C, spec):

    model = pyo.ConcreteModel()

    #
    # sets
    # 
    model.exercises = G.index 
    model.groups = G.columns 
    model.days = np.arange(spec['n_days'])

    #
    # variables 
    #  
    model.v_exercise_schedule = pyo.Var(model.exercises, model.days, domain=pyo.Binary)
    model.v_group_schedule = pyo.Var(model.groups, model.days, domain=pyo.Binary)
    model.v_working_days = pyo.Var(model.days, domain=pyo.Binary)

    #
    # optimization objective
    #
    model.obj = pyo.Objective(expr = sum(model.v_working_days[d] for d in model.days), sense=pyo.minimize)

    #
    # constraints
    #

    #
    # meet the minimum # of sessions for each exercise
    #
    model.c_frequency = pyo.ConstraintList()
    for e in model.exercises:
        model.c_frequency.add(sum(model.v_exercise_schedule[e, d] for d in model.days) == C.loc[e]['count'])
    
    # 
    # translate the exercise schedule into the working days schedule
    #
    model.c_working_days = pyo.ConstraintList()
    for d in model.days:
        n_working_groups = sum(model.v_group_schedule[g, d] for g in model.groups)
        model.c_working_days.add((model.v_working_days[d] * len(model.groups)) >= n_working_groups)
    
    # 
    # translate the exercise schedule into the group schedule
    #
    model.c_working_groups = pyo.ConstraintList()
    for d in model.days:
        for g in model.groups:
            n_working_exercises = sum(model.v_exercise_schedule[e, d] * G.loc[e, g] for e in model.exercises)
            n_group_exercises = sum(G.loc[e,g] for e in model.exercises)
            model.c_working_groups.add((model.v_group_schedule[g, d] * n_group_exercises) >= n_working_exercises)
    
    #
    # ensure that groups meet the required minimum recovery duration
    #
    model.c_recovery = pyo.ConstraintList()
    for g in model.groups:
        min_recovery_days = spec['group_min_recovery_days'][g]
        for k in range(1, min_recovery_days+1):
            for d in model.days:
                next_day = (d+k) % spec['n_days']
                model.c_recovery.add((model.v_group_schedule[g, d] + model.v_group_schedule[g, next_day]) <= 1)
            
    #
    # ensure the number of exercises does not exceed the maximum in each day
    #
    model.c_session_volume = pyo.ConstraintList()
    ed = spec['exercise_durations']
    for d in model.days:
        total_time_sec = sum(model.v_exercise_schedule[e, d] * 
                                  (ed[e]['sets'] * (ed[e]['set_duration_sec'] + ed[e]['rest_period_sec']))
                                  for e in model.exercises)
        model.c_session_volume.add(total_time_sec <= spec['max_session_duration_minutes'] * 60)
    
    return model 

if __name__ == "__main__":
    
    import sys 

    spec_path = sys.argv[1]

    with open(spec_path, 'r') as f:
        spec = json.load(f)

    main(spec)
