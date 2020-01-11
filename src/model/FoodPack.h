#pragma once

#include "Position.h"


class FoodPack
{
public:
	unsigned int food_amount;
	Position position;

	FoodPack(const unsigned int food_amount, const Position position);
	~FoodPack();
};

