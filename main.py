from fastapi import FastAPI
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Tipping Fuzzy Logic"}


@app.get("/tip/{amount}/{food}/{service}")
async def calculate_tip(amount:float, food:float, service:float):

    quality_ant = ctrl.Antecedent(np.arange(0,11,1), 'quality')
    service_ant = ctrl.Antecedent(np.arange(0,11,1), 'service')
    tip_cons = ctrl.Consequent(np.arange(0,26,1), 'tip')

    quality_ant.automf(3)
    service_ant.automf(3)

    tip_cons['low'] = fuzz.trimf(tip_cons.universe, [0,0,13])
    tip_cons['medium'] = fuzz.trimf(tip_cons.universe, [0,13,25])
    tip_cons['high'] = fuzz.trimf(tip_cons.universe, [13,25,25])

    #rules
    rule1 = ctrl.Rule(quality_ant['poor'] | service_ant['poor'], tip_cons['low'])
    rule2 = ctrl.Rule(service_ant['average'],  tip_cons['medium'])
    rule3 = ctrl.Rule(service_ant['good'] | quality_ant['good'], tip_cons['high'])

    #control system
    tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    tipping.input['quality'] = food
    tipping.input['service'] = service

    tipping.compute()
    print(tipping.output['tip'])

    tip_percentage = tipping.output['tip']/100

    return {"tip": round(amount*tip_percentage,2)}
