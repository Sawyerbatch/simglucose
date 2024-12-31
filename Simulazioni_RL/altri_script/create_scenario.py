# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:10:49 2023

@author: Daniele
"""
import numpy as np
import pandas as pd
import json

def create_scenario(n_days, cho_daily=230):

  scenario = []
  cho_sum = 0
  mu_break, sigma_break = 8, 1 
  mu_lunch, sigma_lunch = 13, 1 
  mu_snack, sigma_snack = 17, 1 
  mu_dinner, sigma_dinner = 21, 1 
  mu_night, sigma_night = 24, 1 

  for i in range(n_days):

    mu_cho_break, sigma_cho_break = cho_daily*0.15, 7
    mu_cho_lunch, sigma_cho_lunch = cho_daily*0.45, 20
    mu_cho_snack, sigma_cho_snack = cho_daily*0.05, 2
    mu_cho_dinner, sigma_cho_dinner = cho_daily*0.35, 15
    mu_cho_night, sigma_cho_night = cho_daily*0.05, 2

    hour_break = int(np.random.normal(mu_break, sigma_break/2)) + 24*i
    hour_lunch = int(np.random.normal(mu_lunch, sigma_lunch/2)) + 24*i
    hour_snack = int(np.random.normal(mu_snack, sigma_snack/2)) + 24*i
    hour_dinner = int(np.random.normal(mu_dinner, sigma_dinner/2)) + 24*i
    hour_night = int(np.random.normal(mu_night, sigma_night/2)) + 24*i

    cho_break = int(np.random.normal(mu_cho_break, sigma_cho_break/2))
    cho_lunch = int(np.random.normal(mu_cho_lunch, sigma_cho_lunch/2))
    cho_snack = int(np.random.normal(mu_cho_snack, sigma_cho_snack/2))
    cho_dinner = int(np.random.normal(mu_cho_dinner, sigma_cho_dinner/2))
    cho_night = int(np.random.normal(mu_cho_night, sigma_cho_night/2))

    if int(np.random.randint(100)) < 80:
      scenario.append((hour_break,cho_break))
    if int(np.random.randint(100)) < 100:
      scenario.append((hour_lunch,cho_lunch))
    if int(np.random.randint(100)) < 15: 
      scenario.append((hour_snack,cho_snack))
    if int(np.random.randint(100)) < 95:  
      scenario.append((hour_dinner,cho_dinner))
    if int(np.random.randint(100)) < 1:
      scenario.append((hour_night,cho_night))

    #cho_sum += cho_break + cho_lunch + cho_snack + cho_dinner + cho_night

  return scenario

diz = dict()

ripetizioni = 1000 
giorni = 5

for i in range(ripetizioni):
    scen = create_scenario(giorni, 230)
    diz[i] = scen

with open("scenarios_5_1000_good.json", "w") as outfile:
    json.dump(diz, outfile)