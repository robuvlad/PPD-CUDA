#include "SimulationMap.h"



SimulationMap::SimulationMap(
	const unsigned int height, 
	const unsigned int width, 
	const unsigned int number_ants, 
	Ant * const ants, 
	const unsigned int number_food_packs,
	FoodPack * const foodPacks) :
	height{ height }, width{ width }, number_ants{ number_ants }, ants{ ants }, number_food_packs{ number_food_packs }, foodPacks{ foodPacks }
{
}

SimulationMap::~SimulationMap()
{
}
