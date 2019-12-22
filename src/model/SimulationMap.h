#pragma once

#include "Ant.h"
#include "FoodPack.h"


class SimulationMap
{
public:
	unsigned int height;
	unsigned int width;
	unsigned int number_ants;
	Ant* ants;
	unsigned int number_food_packs;
	FoodPack* foodPacks;

	SimulationMap(const unsigned int height, const unsigned int width, const unsigned int number_ants,  Ant* const ants, const unsigned int number_food_packs, FoodPack* const foodPacks);

	~SimulationMap();
};

