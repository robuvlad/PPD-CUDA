#pragma once

#include "Position.h"


class FoodPack
{
public:
	int food_amount;
	Position position;

	FoodPack(const int food_amount, const Position position);
	~FoodPack();
};

