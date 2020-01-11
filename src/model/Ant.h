#pragma once

#include "Position.h"


class Ant
{
public:
	unsigned int index;
	Position position;
	double speed;
	double direction;
	bool has_food;

	Ant(const unsigned int index, const Position position, const double speed, const double direction);

	~Ant();
};
