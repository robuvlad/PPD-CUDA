#pragma once

#include "Position.h"


class Ant
{
public:
	int index;
	Position position;
	double speed;
	double direction;
	bool has_food;

	Ant(const int index, const Position position, const double speed, const double direction);

	~Ant();
};
