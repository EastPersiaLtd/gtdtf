# Global Terrorism Database with Tensorflow (GTDTF)
## Instruction

GTDTF is based on Global Terrorism Database 1970-2020, which is the biggest terrorism dataset in the world.

To check specific informations, such as predict success, or casulaties, input feature data on simple deep learning model.

## Dataset Information
- Global Terrorism Database 1970-2020(START, 2022)
- 208474 cases
- Test Size=0.4
- Designed to estimate success of terrorism by its feature

## Model Information
1. GTDTFA 1.0 (gtdtfa1_0.r5): Designed to predict success
	- Tensorflow-based
	- 1st layer: 128-nodes TANH
	- 2nd layer: 128-nodes TANH
	- 3rd layer: 1-node SOFTMAX
	- Optimiser: ADAM, LR=0.0001
	- Loss: MSE, 0.1168
	- Accuracy: 88.32%(0.8832)

  
## Model Guide
1. GTDTFA 1.0
	1. iyear: Year of terrorism
	2. imonth: Year of terrorism
	3. region: Region of terrorism
	4. targtype: Target type of terrorism
	5. attacktype: Attack type of terrorism
	6. weaptype: Weapon type of terrorism
	7. extended: 1 - Extended more than 24 hours / 0 - Non-extended
	8. vicinity: 1 - Near of city, urban area / 0 - Rural
	9. suicide: 1 - Suicide terrorism / 0 - Non-sucide terrorism

- To know further details about values, please click the follwing link: [GTD Codebook](*https://www.start.umd.edu/gtd/downloads/Codebook.pdf)

## Quick Guide to predict
```
#For GTDTFA 1.0
#Load the pipeline, scaler, and model
scaler=pickle.load(open('/path/scaler.pkl', "rb"))
loaded_model=tf.keras.models.load_model('/path/gtdtfa1.h5')

#Make predictions on new data
#new_data=[[iyear, imonth, region, targtype, attacktype, weaptype, extended, vicinity, suicide]]
new_data =[[2023, 1, 1, 1, 1, 1, 1, 1, 1]]
new_data_scaled=scaler.transform(new_data)
predictions = loaded_model.predict(new_data_scaled)
print(predictions)
```